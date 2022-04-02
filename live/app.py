from websocket import WebSocketApp
from redis import Redis
import pandas as pd
import json
import talib as ta
import requests

db = Redis(host="db", decode_responses=True)

def on_message(ws, message):
    data = json.loads(message)
    
    ticks = requests.get("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m").json()
    ticks = list(map(lambda x : {"open": x[1], "high": x[2], "low": x[3], "close": x[4]}, ticks))
    historical = pd.DataFrame(ticks)
    
    data["indicators"] = {
        "slope": {
            "open": ta.TAN(historical["open"]).values[-1],
            "high": ta.TAN(historical["high"]).values[-1],
            "low": ta.TAN(historical["low"]).values[-1],
            "close": ta.TAN(historical["close"]).values[-1]
        },
        "rsi": {
            "open": ta.RSI(historical["open"]).values[-1] / 100,
            "high": ta.RSI(historical["high"]).values[-1] / 100,
            "low": ta.RSI(historical["low"]).values[-1] / 100,
            "close": ta.RSI(historical["close"]).values[-1] / 100
        },
        "mom": {
            "open": ta.MOM(historical["open"]).values[-1],
            "high": ta.MOM(historical["high"]).values[-1],
            "low": ta.MOM(historical["low"]).values[-1],
            "close": ta.MOM(historical["close"]).values[-1]
        },
        "patterns": {
            "doji": ta.CDLDOJI(historical["open"], historical["high"], historical["low"], historical["close"]).values[-1] / 100,
            "dojistar": ta.CDLDOJISTAR(historical["open"], historical["high"], historical["low"], historical["close"]).values[-1] / 100,
        }
    }
    db.set("btcusdt", json.dumps(data))        

def on_open(ws):
    print('[**] connection opened [**]')

def on_error(ws, e):
    print(f'[**] exception : {e} [**]')

socket = WebSocketApp(
    "wss://stream.binance.com:9443/ws/btcusdt@kline_1m",
    on_message=on_message,
    on_open=on_open,
    on_error=on_error
)

socket.run_forever()