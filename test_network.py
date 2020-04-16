import numpy as np
import gym
import gym_achtung

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Convolution2D
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

model = load_model('model.h5')
model.test()
