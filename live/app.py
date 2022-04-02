from websocket import WebSocketApp
from redis import Redis

db = Redis(host="db", decode_responses=True)
db.set("start", "NO")

def on_message(ws, message):
    db.set("btcusdt", message)

def on_open(ws):
    print('[**] connection opened [**]')
    
    db.set("start", "YES")

def on_error(ws, e):
    print(f'[**] exception : {e} [**]')

socket = WebSocketApp(
    "wss://stream.binance.com:9443/ws/btcusdt@ticker",
    on_message=on_message,
    on_open=on_open,
    on_error=on_error
)

socket.run_forever()