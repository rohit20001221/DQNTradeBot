import numpy as np
import redis
import json
import random
import time

class TradeEnvironment:
    def __init__(self, margins=10000, quantity=0.00001):
        self.margins = float(margins)
        self.previous_margins = float(margins)
        self.quantity = quantity
        self.initial_margins = float(margins)

        self.positions = set()

        self.db = redis.Redis(host="db")

        self.actions = ["BUY", "SELL", "IDLE"]

    def random(self):
        if "btcusdt" in self.positions:
            output = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            output = random.choice([[1, 0, 0], [0, 0, 1]])

        return output

    def reset(self):
        self.margins = self.initial_margins
        self.positions.clear()

        try:
            data = json.loads(self.db.get("btcusdt"))
        except:
            time.sleep(1)
            return self.reset()

        w = 50000

        o, h, l, c = float(data["k"]["o"]), float(data["k"]["h"]), float(data["k"]["l"]), float(data["k"]["c"])
        rsi = list(data["indicators"]["rsi"].values())
        mom = list(data["indicators"]["mom"].values())
        patterns = list(data["indicators"]["patterns"].values())
        slope = list(data["indicators"]["slope"].values())

        if "btcusdt" in self.positions:
            is_present = 1
        else:
            is_present = 0

        return (np.array(
            [o, h, l, c, is_present, self.margins / self.initial_margins] + rsi + mom + patterns + slope
        ) / w).reshape(1, -1, 1)

    def perform(self, index):
        action = self.actions[index]
        data = json.loads(self.db.get("btcusdt"))

        price = float(data["k"]["c"]) * self.quantity

        if action is "BUY":
            if self.margins > price:
                self.margins -= price

                reward = 0.5
                self.positions.add("btcusdt")
            else:
                reward = 0
        elif action is "SELL":
            if "btcusdt" in self.positions:
                self.margins += price

                self.positions.remove("btcusdt")

                if self.margins > self.previous_margins:
                    reward = 1
                else:
                    reward = -1

                self.previous_margins = self.margins
            else:
                reward = 0
        else:
            reward = 0

        w = 50000

        o, h, l, c = float(data["k"]["o"]), float(data["k"]["h"]), float(data["k"]["l"]), float(data["k"]["c"])
        rsi = list(data["indicators"]["rsi"].values())
        mom = list(data["indicators"]["mom"].values())
        patterns = list(data["indicators"]["patterns"].values())
        slope = list(data["indicators"]["slope"].values())

        if "btcusdt" in self.positions:
            is_present = 1
        else:
            is_present = 0

        print(f"[**] action: {action}, margins: {self.margins}, investment: {self.initial_margins}, previous_margins: {self.previous_margins}, reward: {reward} [**]")        
        return reward, (np.array(
            [o, h, l, c, is_present, self.margins / self.initial_margins] + rsi + mom + patterns + slope
        ) / w).reshape(1, -1, 1)
