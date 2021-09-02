import numpy as np
import gym
import gym_achtung
from gym_achtung.envs.consts import *
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Convolution2D, Conv2D, MaxPool2D, concatenate
from tensorflow.keras.optimizers import Adam, SGD
from datetime import datetime
import time
import random
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import warnings

if __name__ == '__main__':
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

    model = make_model()

    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1500, gamma=0.99,
                   target_model_update=10000,
                   policy=policy)
    dqn.compile(Adam(lr=0.001), metrics=['mse'])
    # Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
    dqn.fit(env, nb_steps=350000, visualize=False, verbose=2)

    trained_model = model.to_json()
    with open("trainedmodel_exp3.json", "w") as file:
        file.write(trained_model)
    file.close()

    dqn.save_weights("model_weights_exp3.h5", overwrite=True)
    print("Saved model to disk")
