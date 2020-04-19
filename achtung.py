import numpy as np
import gym
import gym_achtung
from gym_achtung.envs.consts import *
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Convolution2D, Conv2D, MaxPool2D, concatenate
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

print(env.action_space.n)

# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(256))
# model.add(Activation('relu'))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('relu'))
# print(model.summary())

# def make_conv_layer(input):
#     hidden = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(input)
#     hidden = MaxPool2D((2, 2), padding='same')(hidden)
#     hidden = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(hidden)
#     hidden = MaxPool2D((2, 2), padding='same')(hidden)
#     hidden = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(hidden)
#     hidden = MaxPool2D((2, 2), padding='same')(hidden)
#     return hidden
#
# input1 = Input((1,) + env.observation_space[0].shape)
# conv1 = make_conv_layer(input1)
# input2 = Input((1,) + env.observation_space[1].shape)
# conv2 = make_conv_layer(input2)
#
# conv = concatenate([conv1, conv2])
#
# hidden = Dense(512, activation='relu')(conv)
# hidden = Dense(nb_actions, activation='relu')(hidden)
# hidden = Dense(256, activation='relu')(hidden)
# hidden = Dense(nb_actions, activation='relu')(hidden)
# hidden = Dense(64, activation='relu')(hidden)
# hidden = Dense(nb_actions, activation='relu')(hidden)
# hidden = Dense(16, activation='relu')(hidden)
# main_output = Dense(nb_actions, activation='relu')(hidden)
# model = Model(inputs=[input1, input2], outputs=[main_output])
#
# model.summary()
# --------------------------------------------------------------------


main_input = Input((1,) + env.observation_space.shape)
hidden = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(main_input)
hidden = MaxPool2D((2, 2), padding='same')(hidden)
hidden = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(hidden)
hidden = MaxPool2D((2, 2), padding='same')(hidden)
hidden = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(hidden)
hidden = MaxPool2D((2, 2), padding='same')(hidden)
hidden = Flatten()(hidden)
hidden = Dense(512, activation='relu')(hidden)
main_output = Dense(nb_actions, activation='relu')(hidden)
model = Model(inputs=[main_input], outputs=[main_output])

model.summary()
# --------------------------------------------------------------------

#
# model = Sequential()
# model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(3, 3), input_shape=env.observation_space.shape)) # if soesn't work, consider adding (1,) to the input shape
# model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1)))
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
memory = SequentialMemory(limit=35000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=20000, target_model_update=1e-2,
               policy=policy)
dqn.compile(Adam(lr=0.001), metrics=['mse'])
# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn.fit(env, nb_steps=220000, visualize=False, verbose=2)

trained_model = model.to_json()
with open("trainedmodel.json", "w") as file:
    file.write(trained_model)
file.close()

dqn.save_weights("model_weights.h5", overwrite=True)
model.save("model.h5", overwrite=True)
print("Saved model to disk")

dqn.test(env, nb_episodes=5, visualize=False)
