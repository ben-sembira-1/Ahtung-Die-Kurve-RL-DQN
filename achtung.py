import gym
from tensorflow.keras.optimizers import Adam
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import dqn_agent_manager
import achtung_dqn_models


if __name__ == '__main__':
    manager = dqn_agent_manager.DQNAgentManager(environment=gym.make('achtung-v0'), model_factory=achtung_dqn_models.ConvDense512,
                                                policy=EpsGreedyQPolicy(),
                                                memory=SequentialMemory(limit=50000, window_length=1),
                                                nb_steps_warmup=1500, gamma=0.99, target_model_update=10000,
                                                optimizer=Adam(lr=0.001), metrics=['mse'])

    manager.fit(nb_steps=350000)
    manager.save("test")
