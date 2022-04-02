from bot import DQNAgent, create_model

model = create_model()
agent = DQNAgent(model)

agent.train()