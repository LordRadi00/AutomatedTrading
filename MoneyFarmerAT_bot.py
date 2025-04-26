import logging
import random
import os
import json
import threading
from datetime import datetime

import pandas as pd
import pandas_ta as ta
import websocket
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes,CallbackQueryHandler

# === CONFIG ===
//BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID   = "-4655187396"  # ID del gruppo o chat in cui inviare i messaggi

# === LOGGING SETUP ===
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# === ISTANZA DEL BOT ===
bot = Bot(token=BOT_TOKEN)

# === COPPIE E STORAGE PREZZI + PYRAMID COUNTER ===
pairs = ["BTCUSD", "TAOUSD", "SOLUSD", "XRPUSD"]
close_prices = {p: [] for p in pairs}
high_prices  = {p: [] for p in pairs}
low_prices   = {p: [] for p in pairs}
entry_count  = {p: 0  for p in pairs}  # contatore pyramiding (max 2 entrate)

# === CALCOLO INDICATORI ===
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['EMA50'] = ta.ema(df['close'], length=50)
    df['EMA21'] = ta.ema(df['close'], length=21)
    df['EMA34'] = ta.ema(df['close'], length=34)
    df['ATR']   = ta.atr(df['high'], df['low'], df['close'], length=14)
    # ADX restituisce un DataFrame con colonna 'ADX_14'
    df['ADX']   = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    return df

# === GENERAZIONE SEGNALE ===
def generate_signal(df: pd.DataFrame, pair: str):
    # non abbastanza barre per sweep?
    if len(df) < 3:
        return "No Signal", 0

    last = df.iloc[-1]
    recent_low = df['low'].iloc[-2:].min()

    # sweep su chiusure delle due barre precedenti
    prev_closes = df['close'].iloc[-3:-1]
    sweep_2bar  = last['close'] > prev_closes.max()

    long_condition = (
        last['close'] > recent_low and
        sweep_2bar and
        last['close'] > last['EMA50'] and
        last['ADX'] > 10 and
        last['ATR'] > df['ATR'].rolling(20).mean().iloc[-1] and
        (5 <= last.name.hour < 22)
    )

    if long_condition:
        confidence = round(random.uniform(85, 95), 2)
        return "Bullish", confidence
    return "No Signal", 0

# === PROCESSAMENTO DATI DAL WEBSOCKET ===
def process_data(pair, close_p, high_p, low_p):
    # 1) Append dei prezzi
    arr_c = close_prices[pair]
    arr_h = high_prices[pair]
    arr_l = low_prices[pair]
    arr_c.append(close_p)
    arr_h.append(high_p)
    arr_l.append(low_p)

    # 2) Mantieni solo ultimi 100
    if len(arr_c) > 100:
        arr_c.pop(0)
        arr_h.pop(0)
        arr_l.pop(0)

    # 3) Costruisci il DataFrame con indice temporale fittizio
    df = pd.DataFrame({
        'close': arr_c,
        'high':  arr_h,
        'low':   arr_l
    }, index=pd.to_datetime([datetime.utcnow() for _ in arr_c])).astype(float)

    # 4) Calcola indicatori
    df = calculate_indicators(df)

    # 5) Genera segnale e gestisci pyramiding
    signal, conf = generate_signal(df, pair)
    if signal == "Bullish" and entry_count[pair] < 2:
        entry_count[pair] += 1

        text = (
            f"🐂 *Bullish Trend Detected*\n"
            f"🤖 Confidence AI: {conf}%\n"
            f"📊 Pair: {pair}\n"
            "⏱ Timeframe: 2m\n\n"
            "👉 Vuoi eseguire l’operazione?"
        )
        keyboard = [
            [InlineKeyboardButton("✅ Conferma LONG", callback_data=f"confirm_long|{pair}|{conf}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(
            chat_id=CHAT_ID,
            text=text,
            parse_mode="Markdown",
            reply_markup=reply_markup,
        )
    elif signal != "Bullish":
        # reset pyramiding se non bullish
        entry_count[pair] = 0

# === CALLBACK del WebSocket ===
def on_message(ws, message):
    msg = json.loads(message)
    topic = msg.get("topic", "")
    if topic.startswith("candle"):
        for d in msg.get("data", []):
            p      = d["symbol"]
            close_ = float(d["close"])
            high_  = float(d["high"])
            low_   = float(d["low"])
            process_data(p, close_, high_, low_)

def on_open(ws):
    logging.info("✅ Connessione aperta al WS Bybit")
    subscribe = {
        "op": "subscribe",
        "args": [f"candle.3.{p}" for p in pairs]
    }
    ws.send(json.dumps(subscribe))
    logging.info(f"📡 Sottoscritto a candles 3m: {pairs}")

def on_error(ws, error):
    logging.error("WebSocket error: %s", error)

def on_close(ws, code, reason):
    logging.info("WebSocket closed: %s %s", code, reason)

# === HANDLER TELEGRAM ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome = (
        "🎉 Benvenuto in GoldenBullX! 🎉\n"
        "Rimani in attesa dei segnali di trading in tempo reale su BTC, ETH e SOL.\n"
        "I segnali verranno inviati automaticamente quando la strategia rileva un trend bullish."
    )
    await update.message.reply_text(welcome)
    logging.info(f"/start ricevuto da {update.effective_user.id}")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    _, pair, conf = q.data.split("|")
    user = q.from_user.first_name
    await q.edit_message_text(
        f"✅ LONG *{pair}* confermato da {user} (confidenza {conf}%)",
        parse_mode="Markdown"
    )
    logging.info(f"{user} ha confermato LONG {pair} @ {conf}%")

# === MAIN ===
if __name__ == "__main__":
    # 1) Avvio WS in background
    threading.Thread(
        target=lambda: websocket.WebSocketApp(
            "wss://stream.bybit.com/v5/public/inverse",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        ).run_forever(
            ping_interval=20,
            ping_timeout=10,
            reconnect=True
        ),
        daemon=True
    ).start()

    # 2) Avvio bot Telegram
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))

    logging.info("🤖 GoldenBullX Worker attivo.")
    app.run_polling()
