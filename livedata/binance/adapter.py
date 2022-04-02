from websocket import WebSocketApp
from redis import Redis

class BinanceAdapter:
    def __init__(self, symbol):
        self.ws = WebSocketApp(
            f"wss://stream.binance.com:9443/ws/{symbol}@ticker",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        self.db = Redis()
        
        self.symbol = symbol
    
    def on_open(self, ws):
        print('[**] connection opened [**]')
    
    def on_message(self, ws, message):
        self.db.set(self.symbol, message)
    
    def on_error(self, ws, exception):
        print(f'[**] exception : {exception} [**]')
    
    def on_close(self, ws):
        print('[**] connection closed [**]')
        
    def start(self):
        self.ws.run_forever()
        

if __name__ == '__main__':
    binance = BinanceAdapter("btcusdt")
    binance.start()