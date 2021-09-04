from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D
import time
from datetime import datetime
import random
from rl.agents.dqn import DQNAgent
import warnings
import os


class DQNAgentManager:

    def __init__(self, environment, model_factory, policy, memory, nb_steps_warmup,
                 gamma, target_model_update, optimizer, metrics):
        warnings.simplefilter(action='ignore', category=FutureWarning)

        self.env = environment
        random.seed(int(time.time()))
        # Todo: This may be redundant
        self.env.seed(int(time.time()))
        self.model = model_factory((1,) + self.env.observation_space.shape, self.env.action_space.n)
        self.agent = DQNAgent(model=self.model, nb_actions=self.env.action_space.n, memory=memory,
                              nb_steps_warmup=nb_steps_warmup, gamma=gamma,
                              target_model_update=target_model_update, policy=policy)
        self.agent.compile(optimizer=optimizer, metrics=metrics)

    def fit(self, nb_steps, visualize=False, verbose=2):
        self.agent.fit(self.env, nb_steps=nb_steps, visualize=visualize, verbose=verbose)

    def save(self, name, dir_path='prod/'):
        base_dir = f"{dir_path.rstrip('/')}/{name}"
        dir_created = False
        try:
            os.makedirs(base_dir)
            dir_created = True
        except OSError:
            base_dir += f"_{datetime.now().strftime('%d_%m_%y-%H_%M_%S')}"
        if not dir_created:
            os.makedirs(base_dir)

        # Todo: Maybe create one model and use it all the times.
        with open(f"{base_dir}/model.json", "w") as file:
            file.write(self.model.to_json())
        self.agent.save_weights(f"{base_dir}/weights.h5", overwrite=True)

        print("Saved model to disk")

    def _make_model(self):
        main_input = Input((1,) + self.env.observation_space.shape)
        hidden = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(main_input)
        hidden = MaxPool2D((2, 2), padding='same')(hidden)
        hidden = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(hidden)
        hidden = MaxPool2D((2, 2), padding='same')(hidden)
        hidden = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(hidden)
        hidden = MaxPool2D((2, 2), padding='same')(hidden)
        hidden = Flatten()(hidden)
        hidden = Dense(512, activation='relu')(hidden)
        main_output = Dense(self.env.action_space.n, activation='relu')(hidden)

        return Model(inputs=[main_input], outputs=[main_output])
