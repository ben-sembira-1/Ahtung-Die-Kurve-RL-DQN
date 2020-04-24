import numpy as np
import gym
import gym_achtung
from gym_achtung.envs.consts import *
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Activation, Flatten, Convolution2D, Conv2D, MaxPool2D, concatenate
from keras.optimizers import Adam, SGD

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'achtung-v0'

# Get the environment and extract the number of actions available in the Cartpole problem
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

json_file = open(r'C:\Users\t8763768\PycharmProjects\Ahtung-Die-Kurve-RL-DQN\Saved model json\trainedmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(r'C:\Users\t8763768\PycharmProjects\Ahtung-Die-Kurve-RL-DQN\Saved model weights\model_weights.h5')
print("Loaded model from disk")

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=35000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=2000, target_model_update=1e-2,
               policy=policy)
dqn.compile(Adam(lr=0.001), metrics=['mse'])
# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn.fit(env, nb_steps=220000, visualize=False, verbose=2)

trained_model = model.to_json()
with open("trainedmodelv2.json", "w") as file:
    file.write(trained_model)
file.close()

dqn.save_weights("model_weightsv2.h5", overwrite=True)
model.save("modelv2.h5", overwrite=True)
print("Saved model to disk")

dqn.test(env, nb_episodes=1, visualize=False)
