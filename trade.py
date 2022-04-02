import numpy as np
from livedata.binance.adapter import BinanceAdapter
from livedata.binance.client import BinanceClient
import threading

class TradeEnvironment:
    actions = ["BUY", "SELL", "IDLE"]
        
    def __init__(self, client=BinanceClient(), adapter=BinanceAdapter("btcusdt"), margins=100000, symbol='btcusdt'):
        self.margins = margins
        self.initial_margins = margins
        self.positions = set()
        
        self.symbol = symbol        
        self.client = client
        self.adapter = adapter
        
        threading.Thread(target=self.adapter.start).start()
        
        self.data = self.data_generator()
        
    
    def reset(self):
        self.margins = self.initial_margins
        self.positions.clear()
        
        # return the initial state
        stock_data = next(self.data)
        
        data = np.array(
            stock_data["ohlc"] + [
                stock_data["last_price"],
                stock_data["stock"] in self.positions,
                float(self.margins) / float(self.initial_margins)
            ]
        ).reshape(1, -1, 1)
        
        return data, stock_data

    def perform(self, index, stock_data):
        action = TradeEnvironment.actions[index]
        
        if action == "BUY":
            if stock_data["last_price"] < self.margins:
                self.margins -= stock_data["price"]
                reward = -1 * stock_data["price"]
                self.positions.add(stock_data["stock"])
            else:
                reward = 0
        elif action == "SELL":
            if stock_data["stock"] in self.positions:
                self.margins += stock_data["price"]
                self.positions.remove(stock_data["stock"])
                reward = self.margins - self.initial_margins
            else:
                reward = 0
        else:
            reward = 0
        
        _data = next(self.data)
        
        data = np.array(
            _data["ohlc"] + [
                _data["last_price"],
                _data["stock"] in self.positions,
                float(self.margins) / float(self.initial_margins)
            ]
        ).reshape(1, -1, 1)
        
        return reward / float(self.initial_margins), data, _data
        
    def data_generator(self):
        while True:
            try:
                price = self.client.get(self.symbol)
            except:
                continue
            
            open_price = price.open / price.weighted_average_price
            high_price = price.high / price.weighted_average_price
            low_price = price.low / price.weighted_average_price

            last_price = price.last_price / price.weighted_average_price

            stock = self.symbol

            data = {
                "ohlc": [open_price, high_price, low_price],
                "stock": stock,
                "last_price": last_price,
                "price": price.last_price
            }

            yield data