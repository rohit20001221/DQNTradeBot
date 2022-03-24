# ohlc = np.array([1, 2, 3, 4])
# ohlc = ohlc.reshape((1, -1, 1))
# print(ohlc)
# print(model.predict(ohlc))
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
from collections import deque
import random

# import time

import numpy as np


def generate_random_stock_data():
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


# Deep Q Agent class
class DQNAgent:
    actions = ["BUY", "SELL", "IDLE"]

    def __init__(
        self,
        model: Sequential,
        learning_rate=1e-2,
        gamma=0.8,
        mutation_rate=0.04,
        margins=100000,  # <- initially we have margins of 1 lac by default
        data_generator=generate_random_stock_data(),
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.mutation_rate = mutation_rate
        self.memory = {"states": deque(), "qvalues": deque()}

        self.margins = margins
        self.initial_margin = margins
        self.postions = set()

        self.data_generator = data_generator

    def reset(self):
        self.margins = self.initial_margin
        self.memory["states"].clear()
        self.memory["qvalues"].clear()
        self.postions.clear()

    def train(self, episode_length=1000, num_episodes=10):
        for episode in range(num_episodes):
            # reset the environment before every episode
            self.reset()

            stock_data = next(self.data_generator)

            for transistion in range(episode_length):
                ohlc = stock_data["ohlc"]
                stock = stock_data["stock"]

                if stock in self.postions:
                    is_present = 1
                else:
                    is_present = 0

                data = np.array(
                    ohlc
                    + [
                        stock_data["last_price"],
                        is_present,
                        float(self.margins) / self.initial_margin,
                    ]
                )
                data = data.reshape(1, -1, 1)
                input_state_original = data.reshape(-1, 1)

                if np.random.uniform() < self.mutation_rate:
                    output = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                else:
                    output = list(self.model.predict(data)).pop()

                q_output = np.max(output)
                index = np.argmax(output)

                action = DQNAgent.actions[index]

                # perform the action on the environment
                if action == "BUY":
                    if stock_data["last_price"] < self.margins:
                        self.margins -= stock_data["last_price"]
                        reward = -1 * stock_data["last_price"]
                        self.postions.add(stock)
                elif action == "SELL":
                    if stock in self.postions:
                        self.margins += stock_data["last_price"]
                        self.postions.remove(stock)
                        reward = self.margins - self.initial_margin
                    else:
                        reward = 21
                else:
                    reward = 21

                next_state = next(self.data_generator)

                ohlc = next_state["ohlc"]
                stock = next_state["stock"]

                if stock in self.postions:
                    is_present = 1
                else:
                    is_present = 0

                data = np.array(
                    ohlc
                    + [
                        next_state["last_price"],
                        is_present,
                        float(self.margins) / self.initial_margin,
                    ]
                )
                data = data.reshape(1, -1, 1)

                predection = list(self.model.predict(data)).pop()
                optimal_q = np.max(predection)
                index = np.argmax(predection)

                # use bellman equation to find the updated value of q
                updated_q = (1 - self.learning_rate) * q_output + self.learning_rate * (
                    reward + self.gamma * (optimal_q)
                )

                predection[index] = updated_q
                self.memory["states"].append(input_state_original)
                self.memory["qvalues"].append(predection)

            # train the model
            X = np.array(self.memory["states"])
            print(X)
            # X = np.asarray(X).astype(np.float32)

            Y = np.array(self.memory["qvalues"])
            print(Y)
            # Y = np.asarray(Y).astype(np.float32)

            self.model.fit(X, Y, epochs=10)


if __name__ == "__main__":
    state_space = 7  # open, high, low, close, live_price, is_present, margin_ratio
    action_space = 3  # BUY, SELL, IDLE

    model = Sequential()

    model.add(Dense(64, activation="relu", input_shape=(state_space,)))
    model.add(Dense(64, activation="relu"))

    # output layer total 2 actions
    model.add(Dense(action_space, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy()],
    )

    agent = DQNAgent(model)
    agent.train()
