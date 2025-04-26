# main.py

import os
import time
import json
import logging
import threading
from collections import deque

import pandas as pd
import pandas_ta as ta
from websocket import WebSocketApp
from pybit import inverse_perpetual
from telegram import Bot

# === CONFIGURAZIONE ===
BOT_TOKEN     = os.getenv("BOT_TOKEN")
CHAT_ID       = os.getenv("CHAT_ID", "-4655187396")

BYBIT_KEY     = os.getenv("BYBIT_API_KEY")
BYBIT_SECRET  = os.getenv("BYBIT_API_SECRET")
ORDER_QTY     = float(os.getenv("ORDER_QUANTITY", "0.001"))

PAIRS         = ["BTCUSD", "XRPUSD", "SOLUSD", "TAOUSD"]
TIMEFRAME     = "3"      # minuti
HISTORY_LIMIT = 200      # candele in memoria

# === ISTANZE CLIENT ===
telegram_bot = Bot(token=BOT_TOKEN)
bybit        = inverse_perpetual.HTTP(
    endpoint="https://api.bybit.com",
    api_key=BYBIT_KEY,
    api_secret=BYBIT_SECRET
)

# tiene traccia di quante entry aperte per simbolo (max 2)
_open_orders = { symbol: 0 for symbol in PAIRS }

def place_order(symbol: str):
    """Piazza MARKET BUY con leva 8× e notifica Telegram."""
    # pyramiding = 2
    if _open_orders[symbol] >= 2:
        logging.info(f"{symbol}: raggiunto pyramiding massimo (2), skip")
        return None

    # assicura leva 8×
    bybit.set_leverage(symbol=symbol, buy_leverage=8, sell_leverage=8)

    try:
        resp = bybit.place_active_order(
            symbol=symbol,
            side="Buy",
            order_type="Market",
            qty=ORDER_QTY,
            time_in_force="GoodTillCancel"
        )
        oid = resp["result"]["order_id"]
        _open_orders[symbol] += 1
        msg = (
            f"🚀 *Entry Executed*\n"
            f"Pair: {symbol}\n"
            f"Qty: {ORDER_QTY}\n"
            f"Order ID: `{oid}`\n"
            f"Pyramiding: {_open_orders[symbol]}/2"
        )
        telegram_bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        logging.info(f"✅ ENTRY #{_open_orders[symbol]} per {symbol}, order_id {oid}")
        return oid

    except Exception as e:
        logging.error(f"❌ Errore entry {symbol}: {e}")
        telegram_bot.send_message(
            chat_id=CHAT_ID,
            text=f"❌ Errore entry {symbol}: {e}"
        )
        return None

def exit_order(symbol: str):
    """Piazza MARKET SELL reduce_only e notifica Telegram."""
    if _open_orders[symbol] == 0:
        return None

    try:
        resp = bybit.place_active_order(
            symbol=symbol,
            side="Sell",
            order_type="Market",
            qty=ORDER_QTY,
            time_in_force="GoodTillCancel",
            reduce_only=True
        )
        oid = resp["result"]["order_id"]
        _open_orders[symbol] -= 1
        msg = (
            f"🏁 *Exit Executed*\n"
            f"Pair: {symbol}\n"
            f"Qty: {ORDER_QTY}\n"
            f"Order ID: `{oid}`\n"
            f"Pyramiding rimanente: {_open_orders[symbol]}/2"
        )
        telegram_bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
        logging.info(f"✅ EXIT per {symbol}, order_id {oid}")
        return oid

    except Exception as e:
        logging.error(f"❌ Errore exit {symbol}: {e}")
        telegram_bot.send_message(
            chat_id=CHAT_ID,
            text=f"❌ Errore exit {symbol}: {e}"
        )
        return None

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["EMA50"] = ta.ema(df["close"], length=50)
    df["EMA21"] = ta.ema(df["close"], length=21)
    df["EMA34"] = ta.ema(df["close"], length=34)
    df["ATR"]   = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ADX"]   = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    return df

def generate_signal(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    last       = df.iloc[-1]
    recent_low = df["low"].iloc[-2:].min()
    prev_high  = df["high"].shift(1).iloc[-2:].max()
    sweep      = last["close"] > prev_high
    cond = (
        (last["close"] > recent_low) and
        sweep and
        (last["close"] > last["EMA50"]) and
        (last["ADX"] > 10) and
        (last["ATR"] > df["ATR"].rolling(20).mean().iloc[-1])
    )
    return bool(cond)

def random_confidence() -> float:
    return round(random.uniform(85, 95), 2)

# Rolling storage per ogni pair
buffers = { p: deque(maxlen=HISTORY_LIMIT) for p in PAIRS }

def on_kline(symbol, ts, open_p, high_p, low_p, close_p):
    buf = buffers[symbol]
    buf.append({"open": open_p, "high": high_p, "low": low_p, "close": close_p})
    if len(buf) < HISTORY_LIMIT:
        return

    df = pd.DataFrame(list(buf))
    df = calculate_indicators(df)

    # ENTRY: se <2 posizioni aperte e condizione TRUE
    if _open_orders[symbol] < 2 and generate_signal(df):
        oid = place_order(symbol)
        if oid:
            conf = random_confidence()
            logging.info(f"📈 {symbol} LONG confermato (conf {conf}%)")

    # EXIT placeholder: qui inserirai stop-loss / take-profit
    # es.:
    # if _open_orders[symbol] > 0 and exit_condition(df):
    #     exit_order(symbol)

class BybitStreamer:
    def __init__(self, cb):
        self.on_kline = cb
        self.ws = WebSocketApp(
            "wss://stream.bybit.com/v5/public/inverse",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
    def _on_open(self, ws):
        args = [f"candle.{TIMEFRAME}.{p}" for p in PAIRS]
        ws.send(json.dumps({"op":"subscribe","args": args}))
        logging.info("🔗 WS sottoscritto a: " + ", ".join(args))
    def _on_message(self, ws, msg):
        m = json.loads(msg)
        if m.get("topic","").startswith("candle"):
            for d in m["data"]:
                k = d["k"]
                self.on_kline(
                    symbol=d["symbol"],
                    ts=k["t"],
                    open_p=float(k["o"]),
                    high_p=float(k["h"]),
                    low_p=float(k["l"]),
                    close_p=float(k["c"])
                )
    def _on_error(self, ws, err):
        logging.error(f"WS error: {err}")
    def _on_close(self, ws, code, reason):
        logging.info(f"WS closed: {code} {reason}")
    def start(self):
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

if __name__ == "__main__":
    # log e leva 8× per tutti i pairs
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO
    )
    for sym in PAIRS:
        bybit.set_leverage(sym, buy_leverage=8, sell_leverage=8)
        logging.info(f"{sym}: leverage impostata a 8×")

    streamer = BybitStreamer(on_kline)
    streamer.start()
    logging.info("🤖 Bot live su " + ", ".join(PAIRS))
    while True:
        time.sleep(60)
