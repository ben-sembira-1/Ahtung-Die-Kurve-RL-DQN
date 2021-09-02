import numpy as np
import gym
import gym_achtung
from gym_achtung.envs.consts import *
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Activation, Flatten, Convolution2D, Conv2D, MaxPool2D, concatenate
from keras.optimizers import Adam, SGD
import time
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'achtung-v0'

# Get the environment and extract the number of actions available in the Cartpole problem
env = gym.make(ENV_NAME)
seed = int(time.time())
np.random.seed(seed)
env.seed(seed)
nb_actions = env.action_space.n

exp3_json = r'C:\Users\t8763768\PycharmProjects\Ahtung-Die-Kurve-RL-DQN\trainedmodel_exp3.json'
exp3_weights = r'C:\Users\t8763768\PycharmProjects\Ahtung-Die-Kurve-RL-DQN\model_weights_exp3.h5'

exp4_json = r'C:\Users\t8763768\PycharmProjects\Ahtung-Die-Kurve-RL-DQN\trainedmodel_exp4_continue.json'
exp4_weights = r'C:\Users\t8763768\PycharmProjects\Ahtung-Die-Kurve-RL-DQN\model_weights_exp4_continue.h5'

exp5_json = r'C:\Users\t8763768\PycharmProjects\Ahtung-Die-Kurve-RL-DQN\trainedmodel_exp4_ultra.json'
exp5_weights = r'C:\Users\t8763768\PycharmProjects\Ahtung-Die-Kurve-RL-DQN\model_weights_exp4_ultra.h5'

json_file = open(exp5_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(exp5_weights)
print("Loaded model from disk")

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=300, target_model_update=10000,
               policy=policy)
dqn.compile(Adam(lr=0.001), metrics=['mse'])
# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn.fit(env, nb_steps=200000, visualize=False, verbose=2)

trained_model = model.to_json()
with open("trainedmodel_exp5", "w") as file:
    file.write(trained_model)
file.close()

dqn.save_weights("model_weights_exp5.h5", overwrite=True)
print("Saved model to disk")

dqn.test(env, nb_episodes=5, visualize=False)
print("done!")
