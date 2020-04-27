import numpy as np
import gym
import gym_achtung
from gym_achtung.envs.consts import *
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Convolution2D, Conv2D, MaxPool2D, concatenate
from keras.optimizers import Adam, SGD
from datetime import datetime
import time
import random
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
# seed = int(datetime.now().strftime("%d%H%M%S"))
random.seed(int(time.time()))
env.seed(int(time.time()))
# print("seed:",seed)
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

def make_model():
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
    return model


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

# self, nb_actions, memory, gamma=.99, batch_size=32, nb_steps_warmup=1000,
#                  train_interval=1, memory_interval=1, target_model_update=10000,
#                  delta_range=None, delta_clip=np.inf, custom_model_objects={}, **kwargs):


gamma_test_array = [0.8, 0.85, 0.9, 0.95, 0.99]
warmup_test_array = [1000, 5000, 8000, 13000, 20000]
target_model_update_tset_array = [1e-3, 1e-2, 1e-1, 10000, 20000]
batch_size_array = [32, 64, 128, 512, 1024]

file_names_addons = ["_warmup_", "_gamma_", "_target_model_update_", "_batch_size_"]

tests = [warmup_test_array, gamma_test_array, target_model_update_tset_array, batch_size_array]


def get_parametrs(i, j):
    params = [1000, .99, 1e-3, 32]
    params[i] = tests[i][j]
    return params


for i in range(len(tests)):
    for j in range(len(tests[i])):
        model = make_model()
        policy = EpsGreedyQPolicy()
        params = get_parametrs(i, j)
        memory = SequentialMemory(limit=50000, window_length=1)
        nb_steps_warmup, gamma, target_model_update, batch_size = params[0], params[1], params[2], params[3]
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, gamma=gamma, nb_steps_warmup=nb_steps_warmup,
                       target_model_update=target_model_update,
                       policy=policy, batch_size=batch_size)
        dqn.compile(Adam(lr=0.001), metrics=['mse'])
        # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
        dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

        trained_model = model.to_json()
        path = "trainedjson/trainedmodel_" + str(i) + str(j) + file_names_addons[i] + str(params[i]) + "_.json"
        with open(path, "w") as file:
            file.write(trained_model)
        file.close()

        path = "trainedweights/trainedmodel_weights_" + str(i) + str(j) + file_names_addons[i] + str(params[i]) + "_.h5"
        dqn.save_weights(path, overwrite=True)
        print("Saved model to disk")

        # dqn.test(env, nb_episodes=5, visualize=False)
print("done!")

for i in range(len(tests)):
    for j in range(len(tests[i])):
        model = make_model()
        policy = EpsGreedyQPolicy()
        params = get_parametrs(i, j)
        memory = SequentialMemory(limit=50000, window_length=1)
        nb_steps_warmup, gamma, target_model_update, batch_size = params[0], params[1], params[2], params[3]
        dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, gamma=gamma, nb_steps_warmup=nb_steps_warmup,
                       target_model_update=target_model_update,
                       policy=policy, batch_size=batch_size)
        dqn.compile(Adam(lr=0.01), metrics=['mse'])
        # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
        dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

        trained_model = model.to_json()
        path = "trainedjson/trainedmodel_" + str(i) + str(j) + file_names_addons[i] + str(params[i]) + "_.json"
        with open(path, "w") as file:
            file.write(trained_model)
        file.close()

        path = "trainedweights/trainedmodel_weights_" + str(i) + str(j) + file_names_addons[i] + str(params[i]) + "_.h5"
        dqn.save_weights(path, overwrite=True)
        print("Saved model to disk")

        # dqn.test(env, nb_episodes=5, visualize=False)
print("done!")
