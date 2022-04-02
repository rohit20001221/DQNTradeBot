import numpy as np
import redis
import json

class TradeEnvironment:
    def __init__(self, margins=10000, quantity=0.000001):
        self.margins = float(margins)
        self.quantity = quantity
        self.initial_margins = float(margins)
        
        self.positions = set()

        self.db = redis.Redis(host="db")
        
        self.actions = ["BUY", "SELL", "IDLE"]
        
    def reset(self):
        self.margins = self.initial_margins
        self.positions.clear()
        
        data = json.loads(
            self.db.get("btcusdt")
        )
        
        w = float(data["w"])
        
        o, h, l, c = float(data["o"]), float(data["h"]), float(data["l"]), float(data["c"])
        
        return (np.array(
            [o, h, l, c, "btcusdt" in self.positions]
        ) / w).reshape(1, -1, 1)
    
    def perform(self, index):
        action = self.actions[index]
        data = json.loads(self.db.get("btcusdt"))
        
        price = float(data["c"]) * self.quantity
        
        if action is "BUY":
            if self.margins > price: 
                self.margins -= price
                
                reward = -1 * price * self.quantity
                self.positions.add("btcusdt")
            else:
                reward = 0
        elif action is "SELL":
            if "btcusdt" in self.positions:
                self.margins += price * self.quantity
                
                self.positions.remove("btcusdt")
                reward = (self.margins - self.initial_margins) / self.initial_margins
            else:
                reward = 0
        else:
            reward = 0
        
        w = float(data["w"])
        
        o, h, l, c = float(data["o"]), float(data["h"]), float(data["l"]), float(data["c"])
        
        return reward, (np.array(
            [o, h, l, c, "btcusdt" in self.positions]
        ) / w).reshape(1, -1, 1)
    