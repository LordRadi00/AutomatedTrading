# bybit_quant_bot_ultimate.py
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import telegram
from pybit.unified_trading import HTTP
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import talib

# Configurazione
TELEGRAM_TOKEN = "your_telegram_token"
TELEGRAM_CHAT_ID = "your_chat_id"
BYBIT_API_KEY = "your_api_key"
BYBIT_API_SECRET = "your_api_secret"
SYMBOL = "BTCUSDT"
TIMEFRAME = 15  # minuti
ACCOUNT_TYPE = "UNIFIED"

class AdaptiveTradingBot:
    def __init__(self):
        # Inizializzazione
        self.session = HTTP(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
        self.tg_bot = telegram.Bot(token=TELEGRAM_TOKEN)
        self.model = load_model('models/lstm_model.h5')
        self.scaler = StandardScaler()
        self.equity = self._get_account_balance()
        self.position = None
        self.predictions = []  # Storico predizioni LSTM
        self.trade_log = []

    def _get_account_balance(self):
        res = self.session.get_wallet_balance(accountType=ACCOUNT_TYPE, coin="USDT")
        return float(res['result']['list'][0]['coin'][0]['availableToWithdraw'])

    def fetch_ohlcv(self, limit=500):
        res = self.session.get_kline(
            category="linear" if "FUTURES" in ACCOUNT_TYPE else "spot",
            symbol=SYMBOL,
            interval=TIMEFRAME,
            limit=limit
        )
        df = pd.DataFrame(res['result']['list'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ]).astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.iloc[::-1]

    def preprocess_data(self, df):
        # Indicatori base
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['high'] / df['low'] - 1
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['ma_50'] = df['close'].rolling(50).mean()
        df['ma_200'] = df['close'].rolling(200).mean()
        
        # ATR per gestione del rischio
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Normalizzazione
        features = ['returns', 'volatility', 'rsi', 'obv']
        df[features] = self.scaler.fit_transform(df[features])
        
        return df.dropna()

    def generate_signal(self, df):
        # Preparazione dati LSTM
        seq = df.iloc[-60:][['open', 'high', 'low', 'close', 'volume', 'rsi', 'obv']].values
        lstm_pred = self.model.predict(seq.reshape(1, 60, 7))[0][0]
        
        # Aggiorna storico predizioni
        self.predictions.append(lstm_pred)
        if len(self.predictions) > 100:
            self.predictions.pop(0)
        
        # Soglie dinamiche (media mobile 20 periodi)
        mean_pred = np.mean(self.predictions[-20:]) if len(self.predictions) >= 20 else 0.5
        atr = df['atr'].iloc[-1]
        
        # Condizioni di trading
        long_cond = (lstm_pred > mean_pred + 0.05) and (df['close'].iloc[-1] > df['ma_50'].iloc[-1])
        short_cond = (lstm_pred < mean_pred - 0.05) and (df['close'].iloc[-1] < df['ma_50'].iloc[-1])
        
        if long_cond:
            return 1, atr
        elif short_cond:
            return -1, atr
        return 0, atr

    def execute_trade(self, signal, atr):
        try:
            # Prezzo corrente e dimensionamento
            ticker = self.session.get_tickers(
                category="linear" if "FUTURES" in ACCOUNT_TYPE else "spot",
                symbol=SYMBOL
            )
            price = float(ticker['result']['list'][0]['lastPrice'])
            
            # Calcolo position size basato su ATR
            risk_amount = self.equity * 0.01  # 1% del capitale
            position_size = risk_amount / (1.5 * atr)
            
            # Parametri ordine con SL/TP dinamici
            params = {
                "category": "linear" if "FUTURES" in ACCOUNT_TYPE else "spot",
                "symbol": SYMBOL,
                "side": "Buy" if signal == 1 else "Sell",
                "orderType": "Market",
                "qty": str(round(position_size, 5)),
                "stopLoss": str(price - 1.5 * atr if signal == 1 else price + 1.5 * atr),
                "takeProfit": str(price + 3 * atr if signal == 1 else price - 3 * atr),
                "timeInForce": "GTC"
            }
            
            # Esegui ordine
            response = self.session.place_order(**params)
            
            # Registra posizione
            self.position = {
                'entry_time': datetime.now(),
                'entry_price': price,
                'size': position_size,
                'side': signal,
                'initial_sl': price - 1.5 * atr if signal == 1 else price + 1.5 * atr,
                'tp': price + 3 * atr if signal == 1 else price - 3 * atr,
                'atr': atr,
                'trailing_activated': False
            }
            
            self._send_telegram(
                f"🚀 {'LONG' if signal == 1 else 'SHORT'} {SYMBOL}\n"
                f"Entry: {price:.2f}\n"
                f"Size: {position_size:.4f}\n"
                f"SL: {self.position['initial_sl']:.2f}\n"
                f"TP: {self.position['tp']:.2f}\n"
                f"ATR: {atr:.2f}"
            )
            
        except Exception as e:
            self._send_telegram(f"❌ Errore esecuzione: {str(e)}")

    def monitor_positions(self):
        if not self.position:
            return
            
        try:
            # Prezzo corrente
            ticker = self.session.get_tickers(
                category="linear" if "FUTURES" in ACCOUNT_TYPE else "spot",
                symbol=SYMBOL
            )
            current_price = float(ticker['result']['list'][0]['lastPrice'])
            
            # Timeout dopo 4 ore
            if (datetime.now() - self.position['entry_time']).total_seconds() > 14400:
                self.close_position("TIMEOUT")
                return
                
            # Trailing stop logic
            if self.position['side'] == 1:
                # Attiva trailing dopo 50% del TP
                if not self.position['trailing_activated'] and current_price > self.position['entry_price'] + (self.position['tp'] - self.position['entry_price']) * 0.5:
                    self.position['initial_sl'] = self.position['entry_price'] * 1.002  # Break-even + 0.2%
                    self.position['trailing_activated'] = True
                    self._send_telegram("🔵 Trailing SL attivato (LONG)")
                
                # Aggiorna SL se prezzo sale
                if self.position['trailing_activated']:
                    new_sl = current_price - 1.2 * self.position['atr']  # SL dinamico
                    if new_sl > self.position['initial_sl']:
                        self.position['initial_sl'] = new_sl
                        
                # Check SL/TP
                if current_price <= self.position['initial_sl']:
                    self.close_position("SL")
                elif current_price >= self.position['tp']:
                    self.close_position("TP")
                    
            else:  # Short
                if not self.position['trailing_activated'] and current_price < self.position['entry_price'] - (self.position['entry_price'] - self.position['tp']) * 0.5:
                    self.position['initial_sl'] = self.position['entry_price'] * 0.998  # Break-even - 0.2%
                    self.position['trailing_activated'] = True
                    self._send_telegram("🔵 Trailing SL attivato (SHORT)")
                
                if self.position['trailing_activated']:
                    new_sl = current_price + 1.2 * self.position['atr']
                    if new_sl < self.position['initial_sl']:
                        self.position['initial_sl'] = new_sl
                        
                if current_price >= self.position['initial_sl']:
                    self.close_position("SL")
                elif current_price <= self.position['tp']:
                    self.close_position("TP")
                    
        except Exception as e:
            self._send_telegram(f"⚠️ Errore monitoraggio: {str(e)}")

    def close_position(self, reason):
        try:
            params = {
                "category": "linear" if "FUTURES" in ACCOUNT_TYPE else "spot",
                "symbol": SYMBOL,
                "side": "Sell" if self.position['side'] == 1 else "Buy",
                "orderType": "Market",
                "qty": str(round(self.position['size'], 5))
            }
            response = self.session.place_order(**params)
            exit_price = float(response['result']['price'])
            
            # Calcola PnL
            pnl_pct = ((exit_price - self.position['entry_price']) / 
                      self.position['entry_price'] * 100 * self.position['side'])
            duration = (datetime.now() - self.position['entry_time']).total_seconds() / 60
            
            # Registra trade
            self.trade_log.append({
                'symbol': SYMBOL,
                'side': 'LONG' if self.position['side'] == 1 else 'SHORT',
                'entry': self.position['entry_price'],
                'exit': exit_price,
                'pnl': pnl_pct,
                'duration': duration,
                'reason': reason,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M')
            })
            
            # Notifica
            self._send_telegram(
                f"📌 Chiusura {reason}\n"
                f"PnL: {pnl_pct:.2f}%\n"
                f"Durata: {duration:.1f} min\n"
                f"Saldo: ${self._get_account_balance():.2f}"
            )
            
            # Reset
            self.position = None
            self.equity = self._get_account_balance()
            
        except Exception as e:
            self._send_telegram(f"❌ Errore chiusura: {str(e)}")

    def _send_telegram(self, message):
        try:
            self.tg_bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=f"{datetime.now().strftime('%H:%M:%S')} | {message}"
            )
        except Exception as e:
            print(f"Errore Telegram: {e}")

    def run(self):
        self._send_telegram(f"🤖 Avvio bot {SYMBOL} | Capitale: ${self.equity:.2f}")
        
        try:
            while True:
                start_time = time.time()
                
                # 1. Recupera dati
                df = self.fetch_ohlcv()
                df = self.preprocess_data(df)
                
                # 2. Genera segnale
                signal, atr = self.generate_signal(df)
                
                # 3. Esegui trade (solo se nessuna posizione aperta)
                if not self.position and signal != 0:
                    self.execute_trade(signal, atr)
                
                # 4. Monitora posizioni
                self.monitor_positions()
                
                # 5. Sleep fino al prossimo ciclo
                elapsed = time.time() - start_time
                sleep_time = max(0, (TIMEFRAME * 60) - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self._send_telegram("🛑 Bot fermato manualmente")
        except Exception as e:
            self._send_telegram(f"🔴 Errore critico: {str(e)}")
        finally:
            # Salva log trade
            if self.trade_log:
                pd.DataFrame(self.trade_log).to_csv('trade_log.csv', index=False)

if __name__ == "__main__":
    bot = AdaptiveTradingBot()
    bot.run()
