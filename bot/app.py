from keras.models import load_model
import os
from bot import DQNAgent, create_model

if os.path.exists("/app/checkpoints/model.h5"):
    print("[**] loading previous model [**]")
    model = load_model("/app/checkpoints/model.h5")
else:
    model = create_model()

agent = DQNAgent(model)
agent.train()