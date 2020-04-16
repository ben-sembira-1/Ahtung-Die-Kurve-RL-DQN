import numpy as np
import gym
import gym_achtung
from gym_achtung.envs.consts import *
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D
from keras.optimizers import Adam, SGD

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# TODO: figure out how to give the network a rotated matrix, possible solution- take a larger magtrix, rotate it, and then crop.
# TODO: figure out why the board that the agent sees doesnt include -1 from before
# TODO: check if we should, and if yes how to, fix the network so that it would work with matrices as a CNN
# TODO: remember - let the network totrain A LOTTTT!!!

# TODO: make the lines thicker, and see if cv2.resize works well or how can we process it to be well


ENV_NAME = 'achtung-v0'

# Get the environment and extract the number of actions available in the Cartpole problem
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

print(env.observation_space.shape)

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

#
# model = Sequential()
# model.add(Convolution2D(64, 8, 8, activation='relu', subsample=(4, 4), input_shape= env.observation_space.shape)) # if soesn't work, consider adding (1,) to the input shape
# model.add(Convolution2D(64, 4, 4, activation='relu', subsample=(2, 2)))
# model.add(Convolution2D(32, 3, 3, activation='relu'))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(nb_actions, activation = 'relu'))
# print(model.summary())


'''
another possible model, if the above to slow or that the subsampling makes problems:

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=env.observation_space.shape)) # if doesn't work, consider adding (1,) to the input shape
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(nb_actions, activation = 'relu'))
'''

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=50000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=4000, target_model_update=1e-2,
               policy=policy)
dqn.compile(SGD, metrics=['mse'])

# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn.fit(env, nb_steps=300000, visualize=False, verbose=2)

trained_model = model.to_json()
with open("trainedmodel.json", "w") as file:
    file.write(trained_model)
file.close()

dqn.save_weights("model_weights.h5")
model.save("model.h5")
print("Saved model to disk")

dqn.test(env, nb_episodes=5, visualize=False)
