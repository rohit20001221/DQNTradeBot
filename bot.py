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
import numpy as np
from trade import TradeEnvironment

# Deep Q Agent class
class DQNAgent:
    def __init__(
        self,
        model: Sequential,
        learning_rate=1e-2,
        gamma=0.8,
        mutation_rate=0.04,
        environment=TradeEnvironment()
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.mutation_rate = mutation_rate
        self.memory = {"states": deque(), "qvalues": deque()}
        self.environment = environment

    def reset(self):
        self.memory["states"].clear()
        self.memory["qvalues"].clear()
        
        return self.environment.reset()

    def train(self, episode_length=1000, num_episodes=10):
        for episode in range(num_episodes):
            # reset the environment before every episode
            s, stock_data = self.reset()

            for transistion in range(episode_length):
                if np.random.uniform() < self.mutation_rate:
                    output = random.choice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                else:
                    output = list(self.model.predict(s)).pop()

                q_output = np.max(output)
                index = np.argmax(output)

                reward, s_, _stock_data = self.environment.perform(
                    index, stock_data
                )
                
                optimal_q = np.max(list(self.model.predict(s_)).pop())

                # use bellman equation to find the updated value of q
                updated_q = (1 - self.learning_rate) * q_output + self.learning_rate * (
                    reward + self.gamma * (optimal_q)
                )

                output[index] = updated_q
                self.memory["states"].append(s.reshape(-1, 1))
                self.memory["qvalues"].append(output)
                
                s = s_
                stock_data = _stock_data

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
