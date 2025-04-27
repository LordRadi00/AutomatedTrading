# bybit_auto_trading_pro.py
import os
import time
import base64
import h5py
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pybit.unified_trading import HTTP
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import talib

# 1. Configurazione Bybit
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
SYMBOL = "BTCUSDT"
TIMEFRAME = 15  # minuti
ACCOUNT_TYPE = "UNIFIED"  # Spot o FUTURES

# 2. Parametri Trading
RISK_PER_TRADE = 0.07  # 7% del capitale per trade
STOP_LOSS = 0.015     # 1.5%
TAKE_PROFIT = 0.03    # 3%
COMMISSION = 0.0008   # 0.06% (VIP 1)

class BybitTradingBot:
    def __init__(self):
        self.session = HTTP(
            api_key=API_KEY,
            api_secret=API_SECRET,
            testnet=False
        )
        self.scaler = StandardScaler()
        self._init_models()
        self.equity = self._get_account_balance()
        
    def _init_models(self):
        """Carica modelli pre-addestrati"""
        # LSTM
        self.model = Sequential([
            LSTM(64, input_shape=(60, 7), return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dense(1, activation='sigmoid')
        ])
        self._load_lstm_weights()
        
        # GMM
        self.gmm = self._load_gmm_model()
    
    def _load_lstm_weights(self):
        """Carica pesi LSTM da stringa base64"""
        weights_b64 = "h0AAAAA... [truncated] ...QlRoZW4+"
        weights_bytes = base64.b64decode(weights_b64)
        with h5py.File(io.BytesIO(weights_bytes), 'r') as f:
            self.model.set_weights([f['weight'][:] for weight in f.keys()])
    
    def _load_gmm_model(self):
        """Carica GMM da stringa base64"""
        gmm_b64 = "gASVlQI... [truncated] ...YXVzc2lhbk1peHR1cmWUc2KJlFKUKEsBSwBLAUsCSwN0Yi4="
        return pickle.loads(base64.b64decode(gmm_b64))
    
    def _get_account_balance(self) -> float:
        """Ottieni il saldo disponibile"""
        if ACCOUNT_TYPE == "UNIFIED":
            res = self.session.get_wallet_balance(accountType=ACCOUNT_TYPE, coin="USDT")
            return float(res['result']['list'][0]['coin'][0]['availableToWithdraw'])
        else:
            res = self.session.get_wallet_balance(coin="USDT")
            return float(res['result']['USDT']['available_balance'])
    
    def fetch_ohlcv(self, limit: int = 1000) -> pd.DataFrame:
        """Ottieni i dati OHLCV più recenti"""
        res = self.session.get_kline(
            category="linear" if "FUTURES" in ACCOUNT_TYPE else "spot",
            symbol=SYMBOL,
            interval=TIMEFRAME,
            limit=limit
        )
        df = pd.DataFrame(res['result']['list'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        df = df.astype({
            'open': float, 'high': float, 'low': float,
            'close': float, 'volume': float
        })
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df.iloc[::-1].reset_index(drop=True)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering completo"""
        # Features base
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['high'] / df['low'] - 1
        df['volume_z'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
        
        # Indicatori avanzati
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
        
        # Normalizzazione
        features = ['returns', 'volatility', 'volume_z', 'rsi', 'obv', 'adx']
        df[features] = self.scaler.fit_transform(df[features])
        
        return df.dropna()
    
    def detect_regime(self, df: pd.DataFrame) -> int:
        """Classifica regime di mercato"""
        features = df[['returns', 'volatility', 'volume_z']].values[-100:]
        return self.gmm.predict(features[-1].reshape(1, -1))[0]
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[int]:
        """Genera segnale di trading"""
        # Preparazione sequenza LSTM
        seq = df.iloc[-60:][['open', 'high', 'low', 'close', 'volume', 'rsi', 'obv']].values
        lstm_prob = self.model.predict(seq.reshape(1, 60, 7))[0][0]
        
        # Regole di entrata
        long_cond = (df['close'].iloc[-1] > df['close'].rolling(50).mean().iloc[-1]) and (lstm_prob > 0.7)
        short_cond = (df['close'].iloc[-1] < df['close'].rolling(50).mean().iloc[-1]) and (lstm_prob < 0.3)
        
        if long_cond:
            return 1  # LONG
        elif short_cond:
            return -1  # SHORT
        return None
    
    def place_order(self, signal: int):
        """Esegui ordine con gestione del rischio"""
        price = self.session.get_tickers(category="linear" if "FUTURES" in ACCOUNT_TYPE else "spot", symbol=SYMBOL)['result']['list'][0]['lastPrice']
        price = float(price)
        
        # Calcola size posizione
        position_size = (self.equity * RISK_PER_TRADE) / price
        
        # Parametri ordine
        params = {
            "category": "linear" if "FUTURES" in ACCOUNT_TYPE else "spot",
            "symbol": SYMBOL,
            "side": "Buy" if signal == 1 else "Sell",
            "orderType": "Market",
            "qty": str(round(position_size, 5)),
            "timeInForce": "GTC",
            "stopLoss": str(price * (1 - STOP_LOSS)) if signal == 1 else str(price * (1 + STOP_LOSS)),
            "takeProfit": str(price * (1 + TAKE_PROFIT)) if signal == 1 else str(price * (1 - TAKE_PROFIT))
        }
        
        # Invia ordine
        try:
            response = self.session.place_order(**params)
            print(f"Ordine eseguito: {response}")
            return True
        except Exception as e:
            print(f"Errore nell'ordine: {e}")
            return False
    
    def run(self):
        """Esegui il loop di trading"""
        print(f"Starting bot with ${self.equity:.2f} balance")
        
        while True:
            try:
                # 1. Ottieni dati
                df = self.fetch_ohlcv(limit=500)
                df = self.preprocess_data(df)
                
                # 2. Genera segnale
                signal = self.generate_signal(df)
                
                # 3. Esegui trade
                if signal is not None:
                    print(f"Segnale generato: {'LONG' if signal == 1 else 'SHORT'}")
                    self.place_order(signal)
                
                # 4. Aggiorna saldo
                self.equity = self._get_account_balance()
                print(f"Saldo attuale: ${self.equity:.2f}")
                
                # 5. Aspetta il prossimo candle
                time.sleep(TIMEFRAME * 60 - (time.time() % (TIMEFRAME * 60)))
                
            except Exception as e:
                print(f"Errore: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = BybitTradingBot()
    bot.run()
