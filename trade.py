import numpy as np
import random

class TradeEnvironment:
    actions = ["BUY", "SELL", "IDLE"]
        
    def __init__(self, margins=100000):
        self.margins = margins
        self.initial_margins = margins
        self.positions = set()
        
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
                self.margins -= stock_data["last_price"]
                reward = -1 * stock_data["last_price"]
                self.positions.add(stock_data["stock"])
        elif action == "SELL":
            if stock_data["stock"] in self.positions:
                self.margins += stock_data["last_price"]
                self.positions.remove(stock_data["stock"])
                reward = self.margins - self.initial_margins
            else:
                reward = 21
        else:
            reward = 21
        
        _data = next(self.data)
        
        data = np.array(
            _data["ohlc"] + [
                _data["last_price"],
                _data["stock"] in self.positions,
                float(self.margins) / float(self.initial_margins)
            ]
        ).reshape(1, -1, 1)
        
        return reward, data, _data
        
    def data_generator(self):
        stocks = ["A", "B", "C", "D", "E"]

        while True:
            open_price = np.random.uniform()
            high_price = np.random.uniform()
            low_price = np.random.uniform()
            close_price = np.random.uniform()

            last_price = np.random.uniform()

            stock = random.choice(stocks)

            data = {
                "ohlc": [open_price, high_price, low_price, close_price],
                "stock": stock,
                "last_price": last_price,
            }

            yield data