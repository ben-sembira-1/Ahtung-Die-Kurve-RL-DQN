"""
imports:
"""
# from ..gym-achtung.gym_achtung.envs.consts

import numpy as np
import gym
import gym_achtung

from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'achtung-v0'
# Get the environment and extract the number of actions available in the Cartpole problem
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
# nb_observations = env.observation_space.n

print(env.observation_space.shape)

input = Input(shape=env.observation_space.shape)
x = Flatten()(input)
x = Dense(16, activation='relu')(x)
x = Dense(16, activation='relu')(x)
x = Dense(16, activation='relu')(x)
output = Dense(nb_actions, activation='sigmoid')(x)
model = Model(inputs=input, outputs=output)

# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) #if there is an error consider delete: (1,) +
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('linear'))
print(model.summary())

policy = EpsGreedyQPolicy(eps=0.1)
memory = SequentialMemory(limit=50000, window_length=1)
dqn_agent = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, target_model_update=1e-2,
                     policy=policy)
dqn_agent.compile(Adam(lr=1e-3), metrics=['mae'])
# Okay, now it's time to learn something! We visualize the training here for show, but this slows down training quite a lot.
dqn_agent.fit(env, nb_steps=5000, visualize=False, verbose=2)

# dqn_agent.test(env, EPISODES, visualize=True)

# Iterate the game
# for e in range(EPISODES):
#
#     # reset state in the beginning of each game
#     state = env.reset()
#     state = np.reshape(state, [1, 4])
#     ################################################################
#     # REMEMBER TO CHANGE THE RESHAPE WHEN ENV.RESET IS DEFINED
#     ################################################################
#
#     # time_t represents each frame of the game
#     # Our goal is to keep the pole upright as long as possible until score of 500
#     # the more time_t the more score
#     for time_t in range(NUMBER_OF_TURNS_TO_LIVE):
#
#         # turn this on if you want to render
#         # env.render()
#
#         # Decide action
#         action = dqn_agent.act(state)
#
#         # Advance the game to the next frame based on the action.
#         # Reward is 1 for every frame the pole survived
#         next_state, reward, done, _ = env.step(action)
#         next_state = np.reshape(next_state, [1, 4])
#         ################################################################
#         # REMEMBER TO CHANGE THE RESHAPE WHEN ENV.RESET IS DEFINED
#         ################################################################
#
#         # memorize the previous state, action, reward, and done
#         dqn_agent.memorize(state, action, reward, next_state, done)
#         #IF ABOVE DOESNT WORK CONSIDER REPLACE MEMORIZE WITH REMEMBER
#
#         # make next_state the new current state for the next frame.
#         dqn_agent.replay()
#         dqn_agent.target_train()
#
#         state = next_state
#
#         # done becomes True when the game ends
#         # ex) The agent drops the pole
#         # if game_over:
#         #     # print the score and break out of the loop
#         #     print("EPISODESS: {}/{}, score: {}"
#         #           .format(time, EPISODESs, time_t))
#         #
#         #     break
#
#     # train the agent with the experience of the EPISODES


# def get_action(self, states):
#         if np.random.random() < DQN_EPSIOLN:
#             return np.random.choice(env.action_space)
#         else:
#             return np.argmax(self.predict())

#
# def create_model(self):
#     model = Sequential()
#     state_shape = self.env.observation_space.shape
#     model.add(Dense(24, input_dim=state_shape[0],
#                     activation="relu"))
#     model.add(Dense(48, activation="relu"))
#     model.add(Dense(24, activation="relu"))
#     model.add(Dense(self.env.action_space.n))
#     model.compile(loss="mean_squared_error",
#                   optimizer=Adam(lr=self.learning_rate))
#     return model
