import numpy as np
import gym
import gym_achtung
from gym_achtung.envs.consts import *
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Activation, Flatten, Convolution2D, Conv2D, MaxPool2D, concatenate
import keras.optimizers

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'achtung-v0'

if __name__ == '__main__':
    # Get the environment and extract the number of actions available in the Cartpole problem
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    json_file = open(r'trainedjson/trainedmodel_00_warmup_1000_.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(r'trainedweights/trainedmodel_weights_00_warmup_1000_.h5')
    print("Loaded model from disk")

    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=35000, window_length=1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=2000, target_model_update=1e-2,
                   policy=policy)
    dqn.compile(keras.optimizers.get('adam')(lr=0.001), metrics=['mse'])
    # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.

    dqn.test(env, nb_episodes=20, visualize=True)
