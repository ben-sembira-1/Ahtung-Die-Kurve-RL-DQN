import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_achtung.envs.ahtungGame import *
import pygame
import numpy as np
from gym_achtung.envs.consts import *

# import skimage


class AchtungEnv(gym.Env):
    networks = []
    metadata = {'render.modes': ['human']}
    number_of_players = 1

    def __init__(self):
        self.game = AchtungGame(AchtungEnv.number_of_players)
        self.players = [self.game.get_player_by_id(id + 1) for id in range(AchtungEnv.number_of_players)]
        self.action_space = spaces.Discrete(3)
        self.scaling = SCALING
        self.observation_space = spaces.MultiDiscrete((SCREEN_WIDTH // self.scaling) * (SCREEN_HEIGHT // self.scaling))
        self.seed()

    '''

            ##high = np.array(
                [SCREEN_WIDTH // self.scaling, SCREEN_HEIGHT // self.scaling])  # related to the 2-nd method below
            ##low = np.zeros(2)

            one_player_location_space = spaces.box(low, high, dtype=np.int16)
            # high = np.array([SCREEN_WIDTH//self.scaling, SCREEN_HEIGHT//self.scaling]*FooEnv.number_of_players)
            # low = np.zeros(2 * FooEnv.number_of_players)
            # all_players_location_space = spaces.box(low, high, dtype=np.int16)

            # 0,1 1d array in size SCREEN_WIDTH/t X SCREEN_HEIGHT/t
            ## game_board_space = spaces.multi_binary(SCREEN_WIDTH//self.scaling * SCREEN_HEIGHT//self.scaling)
            # 0,1 1d array in size SCREEN_WIDTH/t X SCREEN_HEIGHT/t box of all players with dtype=np.int16
            ## self.observation_space = spaces.tuple(one_player_location_space, game_board_space)
    

    # 0,1 1d array in size SCREEN_WIDTH/t X SCREEN_HEIGHT/t for some t, box of player with dtype=np.int16
    # related to the above method: self.observation_space = spaces.tuple(one_player_location_space, game_board_space)

    # 0,1 1d array in size 2*SCREEN_WIDTH/t X 2*SCREEN_HEIGHT/t player allways in center
    '''

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):  # actions = the output of each network, the actions they want to do

        # resized_game_board = skimage.measure.block_reduce(self.game.game_board, (SCREEN_RESIZE_FACTOR,SCREEN_RESIZE_FACTOR), np.max)
        # resized_game_board = np.array([[1 if x > 0 else 0 for x in row] for row in resized_game_board])

        actions_to_send = [(self.players[0], actions[0])]
        # get action from a random trained network
        for i in range(1, AchtungEnv.number_of_players):
            chosen_network = np.random.choice(AchtungEnv.networks)
            pred = chosen_network.predict(self.get_state(id=i + 1))
            max = (pred[0], 0)
            for i in range(len(pred)):
                if pred[i] > max[0]:
                    max = (pred[i], i)
            action_from_trained_network = max[0] - 1
            actions_to_send.append((self.players[i], action_from_trained_network))

        # do move
        board_not_processed, players, game_over = self.game.step(actions_to_send)
        reward = self.get_reward(game_over, players)

        return self.get_state(), reward, bool(game_over), {}

    def get_state(self, id=1):
        resized_game_board = self.preprocess_board(self.game.game_board)
        curr_place = (self.players[id-1].x, self.players[id-1].y)
        return (curr_place, resized_game_board)

    def preprocess_board(self, board):
        new_board = np.zeros(
            (
                len(board)//self.scaling,
                len(board[0])//self.scaling
            )
        )

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j]:
                    new_board[i//self.scaling][j//self.scaling] = 1

        return new_board

    def get_reward(self, game_over, players):
        '''
        Reward with many pklayers in the game
        if not game_over and self.players[0] in players:
            return -len(players)
        elif game_over and self.players[0] in players:
            return WIN_REWARD
            '''
        if not game_over and self.players[0] in players:
            return 1
        else:
            print('Game is over!\nThe network played ' + str(self.game.num_of_turns) + 'turns.')

    def reset(self):
        self.game = AchtungGame(AchtungEnv.number_of_players)
        self.players = [self.game.get_player_by_id(id + 1) for id in range(AchtungEnv.number_of_players)]
        self.seed()

    def render(self, mode='human', close=False):
        pass
