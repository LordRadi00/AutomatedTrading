# bybit_quant_bot_pro.py
import os
import time
import base64
import h5py
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pybit.unified_trading import HTTP, WebSocket
import telegram
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorforce.agents import Agent
from tensorforce.environments import Environment
import talib

# 1. Configurazione
TELEGRAM_TOKEN = "your_telegram_token"
TELEGRAM_CHAT_ID = "your_chat_id"
BYBIT_API_KEY = "your_api_key"
BYBIT_API_SECRET = "your_api_secret"
SYMBOL = "BTCUSDT"
TIMEFRAME = 15  # minuti
ACCOUNT_TYPE = "UNIFIED"

# 2. Parametri Trading
RISK_PER_TRADE = 0.1
STOP_LOSS = 0.015
TAKE_PROFIT = 0.03
COMMISSION = 0.0006

class TradingEnvironment(Environment):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.reset()

    def states(self):
        return dict(type='float', shape=(10,))

    def actions(self):
        return dict(type='int', num_values=3)  # 0=hold, 1=long, 2=short

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def execute(self, actions):
        signal = actions - 1  # Convert to -1,0,1
        reward = self.bot.execute_rl_trade(signal)
        next_state = self._next_observation()
        done = False
        return next_state, done, reward

    def _next_observation(self):
        df = self.bot.fetch_ohlcv(100)
        features = self.bot.extract_features(df).iloc[-1].values
        return features[:10]  # First 10 features

class BybitQuantBot:
    def __init__(self):
        # Inizializza connessioni
        self.session = HTTP(
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
            testnet=False
        )
        self.ws = WebSocket(
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
            testnet=False
        )
        self.tg_bot = telegram.Bot(token=TELEGRAM_TOKEN)
        
        # Inizializza modelli
        self.scaler = StandardScaler()
        self._init_models()
        self.equity = self._get_account_balance()
        self.position = None
        self.rl_agent = Agent.create(
            agent='ppo',
            environment=TradingEnvironment(self),
            network='auto',
            batch_size=10,
            learning_rate=1e-3
        )

    def _init_models(self):
        """Carica tutti i modelli ML"""
        # LSTM
        self.model = Sequential([
            LSTM(64, input_shape=(60, 9),  # +2 features order book
            Dropout(0.3),
            LSTM(32),
            Dense(1, activation='sigmoid')
        ])
        self._load_lstm_weights()
        
        # GMM
        self.gmm = self._load_gmm_model()

    def _load_lstm_weights(self):
        weights_b64 = "h0AAAAA... [truncated] ...QlRoZW4+"
        weights_bytes = base64.b64decode(weights_b64)
        with h5py.File(io.BytesIO(weights_bytes), 'r') as f:
            self.model.set_weights([f['weight'][:] for weight in f.keys()])

    def _load_gmm_model(self):
        gmm_b64 = "gASVlQI... [truncated] ...YXVzc2lhbk1peHR1cmWUc2KJlFKUKEsBSwBLAUsCSwN0Yi4="
        return pickle.loads(base64.b64decode(gmm_b64))

    def send_telegram(self, message: str):
        """Invia notifica su Telegram"""
        try:
            self.tg_bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            print(f"Errore Telegram: {e}")

    def _get_account_balance(self) -> float:
        res = self.session.get_wallet_balance(accountType=ACCOUNT_TYPE, coin="USDT")
        return float(res['result']['list'][0]['coin'][0]['availableToWithdraw'])

    def fetch_ohlcv(self, limit: int = 500) -> pd.DataFrame:
        res = self.session.get_kline(
            category="linear" if "FUTURES" in ACCOUNT_TYPE else "spot",
            symbol=SYMBOL,
            interval=TIMEFRAME,
            limit=limit
        )
        df = pd.DataFrame(res['result']['list'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ]).astype({
            'open': float, 'high': float, 'low': float,
            'close': float, 'volume': float
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.iloc[::-1]

    def get_orderbook(self) -> Tuple[float, float]:
        """Ottieni bid/ask top e liquidity"""
        res = self.session.get_orderbook(
            category="linear" if "FUTURES" in ACCOUNT_TYPE else "spot",
            symbol=SYMBOL
        )
        bid = float(res['result']['b'][0][0])
        ask = float(res['result']['a'][0][0])
        bid_size = float(res['result']['b'][0][1])
        ask_size = float(res['result']['a'][0][1])
        return bid, ask, bid_size, ask_size

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering avanzato con order book"""
        # Features tecniche
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['high'] / df['low'] - 1
        df['volume_z'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Aggiungi dati order book
        bid, ask, bid_size, ask_size = self.get_orderbook()
        df['spread'] = (ask - bid) / ask
        df['liq_imbalance'] = (bid_size - ask_size) / (bid_size + ask_size)
        
        # Normalizzazione
        features = ['returns', 'volatility', 'volume_z', 'rsi', 'obv', 'spread', 'liq_imbalance']
        df[features] = self.scaler.fit_transform(df[features])
        
        return df.dropna()

    def detect_regime(self, df: pd.DataFrame) -> int:
        features = df[['returns', 'volatility', 'volume_z', 'spread']].values[-100:]
        return self.gmm.predict(features[-1].reshape(1, -1))[0]

    def generate_signal(self, df: pd.DataFrame) -> Optional[int]:
        """Genera segnale ibrido"""
        # 1. Predizione LSTM
        seq = df.iloc[-60:][['open', 'high', 'low', 'close', 'volume', 'rsi', 'obv', 'spread', 'liq_imbalance']].values
        lstm_prob = self.model.predict(seq.reshape(1, 60, 9))[0][0]
        
        # 2. RL Agent
        state = self.rl_agent.environment.reset()
        rl_action = self.rl_agent.act(state)
        rl_signal = rl_action - 1
        
        # 3. Regole di fusione
        long_cond = (lstm_prob > 0.7) and (df['close'].iloc[-1] > df['close'].rolling(50).mean().iloc[-1])
        short_cond = (lstm_prob < 0.3) and (df['close'].iloc[-1] < df['close'].rolling(50).mean().iloc[-1])
        
        # Decisione finale
        if rl_signal == 1 and long_cond:
            return 1
        elif rl_signal == -1 and short_cond:
            return -1
        return None

    def execute_trade(self, signal: int):
        """Esegui trade con gestione avanzata"""
        # Prezzi e size
        bid, ask, bid_size, ask_size = self.get_orderbook()
        price = ask if signal == 1 else bid
        position_size = (self.equity * RISK_PER_TRADE) / price
        
        # Parametri ordine
        params = {
            "category": "linear" if "FUTURES" in ACCOUNT_TYPE else "spot",
            "symbol": SYMBOL,
            "side": "Buy" if signal == 1 else "Sell",
            "orderType": "Limit",
            "qty": str(round(position_size, 5)),
            "price": str(price),
            "timeInForce": "PostOnly",
            "stopLoss": str(price * (1 - STOP_LOSS)) if signal == 1 else str(price * (1 + STOP_LOSS)),
            "takeProfit": str(price * (1 + TAKE_PROFIT)) if signal == 1 else str(price * (1 - TAKE_PROFIT))
        }
        
        try:
            # Esegui ordine
            response = self.session.place_order(**params)
            self.position = {
                'entry_price': price,
                'size': position_size,
                'side': signal,
                'time': datetime.now()
            }
            
            # Notifica Telegram
            msg = f"""🚀 NUOVA OPERAZIONE
{'LONG' if signal == 1 else 'SHORT'} {SYMBOL}
Entry: {price:.2f}
Size: {position_size:.4f} BTC
SL: {params['stopLoss']}
TP: {params['takeProfit']}"""
            self.send_telegram(msg)
            
            return True
        except Exception as e:
            self.send_telegram(f"❌ ERRORE ORDINE: {str(e)}")
            return False

    def execute_rl_trade(self, signal: int) -> float:
        """Esegui trade per RL environment"""
        if signal == 0:  # Hold
            return 0.0
            
        executed = self.execute_trade(signal)
        if not executed:
            return -0.1  # Penalità per errore
            
        # Calcola reward dopo chiusura trade
        while self.position is not None:
            time.sleep(60)
            
        # Reward basato su PnL
        pnl = self.position.get('pnl_pct', 0)
        return pnl * 10  # Scalato per training

    def monitor_position(self):
        """Monitora posizione aperta"""
        if self.position is None:
            return
            
        current_price = float(self.session.get_tickers(
            category="linear" if "FUTURES" in ACCOUNT_TYPE else "spot",
            symbol=SYMBOL
        )['result']['list'][0]['lastPrice'])
        
        # Check SL/TP
        if (self.position['side'] == 1 and current_price <= float(self.position['stop_loss'])) or \
           (self.position['side'] == -1 and current_price >= float(self.position['stop_loss'])):
            self.close_position("SL")
        elif (self.position['side'] == 1 and current_price >= float(self.position['take_profit'])) or \
             (self.position['side'] == -1 and current_price <= float(self.position['take_profit'])):
            self.close_position("TP")

    def close_position(self, reason: str):
        """Chiudi posizione corrente"""
        try:
            # Chiudi posizione
            params = {
                "category": "linear" if "FUTURES" in ACCOUNT_TYPE else "spot",
                "symbol": SYMBOL,
                "side": "Sell" if self.position['side'] == 1 else "Buy",
                "orderType": "Market",
                "qty": str(self.position['size'])
            }
            response = self.session.place_order(**params)
            
            # Calcola PnL
            exit_price = float(response['result']['price'])
            pnl_pct = (exit_price - self.position['entry_price']) / self.position['entry_price'] * 100
            pnl_pct *= self.position['side']
            
            # Notifica
            msg = f"""📌 CHIUSURA POSIZIONE
Motivo: {reason}
{'LONG' if self.position['side'] == 1 else 'SHORT'} {SYMBOL}
Entry: {self.position['entry_price']:.2f}
Exit: {exit_price:.2f}
Durata: {(datetime.now() - self.position['time']).total_seconds()/60:.1f}m
PnL: {pnl_pct:.2f}%"""
            self.send_telegram(msg)
            
            # Aggiorna stato
            self.position['pnl_pct'] = pnl_pct
            self.position = None
            self.equity = self._get_account_balance()
            
            return True
        except Exception as e:
            self.send_telegram(f"❌ ERRORE CHIUSURA: {str(e)}")
            return False

    def run(self):
        """Main trading loop"""
        self.send_telegram(f"🤖 Bot avviato con capitale: ${self.equity:.2f}")
        
        try:
            while True:
                start_time = time.time()
                
                # 1. Aggiorna dati
                df = self.fetch_ohlcv()
                df = self.extract_features(df)
                self.monitor_position()
                
                # 2. Genera segnale (solo se nessuna posizione aperta)
                if self.position is None:
                    signal = self.generate_signal(df)
                    if signal is not None:
                        self.execute_trade(signal)
                
                # 3. Addestra RL agent
                if len(df) > 100:
                    self.rl_agent.observe(reward=0)
                    self.rl_agent.act(states=self.rl_agent.environment.reset())
                
                # 4. Sleep fino al prossimo ciclo
                elapsed = time.time() - start_time
                sleep_time = max(0, (TIMEFRAME * 60) - elapsed)
                time.sleep(sleep_time)
                
        except Exception as e:
            self.send_telegram(f"🔴 ERRORE CRITICO: {str(e)}")
            raise

if __name__ == "__main__":
    bot = BybitQuantBot()
    bot.run()
