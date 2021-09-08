import gym
from gym import error, spaces, utils
from gym.utils import seeding
# import pygame
import numpy as np
import matplotlib.pyplot as plt
from .consts import *
import colorama
from colorama import Fore, Style
import cv2
from gym_achtung.envs.ahtungGame import *
# import skimage
# Todo: png was not commented, but a research for wat package is it is required.
# import png
from scipy import ndimage


class AchtungEnv(gym.Env):
    networks = []
    metadata = {'render.modes': ['human']}
    number_of_players = NUMBER_OF_PLAYERS

    # file_to_print_matrix = open("game.txt", "w")

    def __init__(self):
        # ------ pygame ------
        # pygame.init()
        # self.screen = pygame.display.set_mode([SCREEN_WIDTH // 4, SCREEN_HEIGHT // 4])
        # self.screen.fill((0, 0, 0))

        # simulation:
        self.reward = 0
        self.game = AchtungGame(AchtungEnv.number_of_players)
        self.players = [self.game.get_player_by_id(id + 1) for id in range(AchtungEnv.number_of_players)]
        self.action_space = spaces.Discrete(3)
        # self.observation_board_width = SCREEN_WIDTH//(SCALING * ZOOM) + 1
        # self.observation_board_height = SCREEN_HEIGHT//(SCALING * ZOOM) + 1

        # self.observation_board_width = SCREEN_WIDTH//(ZOOM)
        # self.observation_board_height = SCREEN_HEIGHT//(ZOOM)

        self.observation_board_width = FINAL_SIZE
        self.observation_board_height = FINAL_SIZE

        # self.observation_space_size = (self.observation_board_width * self.observation_board_height)

        # high = np.array([2] * self.observation_space_size + [1, 1])
        # low = np.array([-1] * self.observation_space_size + [-1, -1])
        # high = np.array([2] * self.observation_space_size)
        # low = np.array([-1] * self.observation_space_size)
        high = np.array([2] * (FINAL_SIZE ** 2) + [1, 1])
        low = np.array([-1] * (FINAL_SIZE ** 2) + [-1, -1])
        # self.observation_space = spaces.Box(low, high, dtype=np.int16)
        # self.observation_space = np.zeros((2, self.observation_board_height, self.observation_board_width))
        self.observation_space = np.zeros((self.observation_board_height, self.observation_board_width))
        self.num_of_steps = 0

        # self.observation_space = spaces.MultiDiscrete((SCREEN_WIDTH // SCALING) * (SCREEN_HEIGHT // SCALING))
        self.seed()
        # self.img_memory = np.zeros((self.observation_board_height, self.observation_board_width))

        # TODO: TORUN
        # ----------------------------------------
        # self.observation_board_width = FINAL_SIZE
        # self.observation_board_height = FINAL_SIZE
        # high = np.array([2] * (FINAL_SIZE ** 2) + [1, 1])
        # low = np.array([-1] * (FINAL_SIZE ** 2) + [-1, -1])
        # self.observation_space = spaces.Box(low, high, dtype=np.int16)
        # self.num_of_steps = 0
        # self.seed()
        # ----------------------------------------
        # printing board with plt

        #    -------- network --------
        exp3_model = r'F:\Projects\Achtung Die Kurve\Dev\models\Good Models\trainedmodel_exp_rotate.json'
        exp3_weights = r'F:\Projects\Achtung Die Kurve\Dev\models\Good Models\model_weights_exp_rotate.h5'

        json_file = open(exp3_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.net = model_from_json(loaded_model_json)
        self.net.load_weights(exp3_weights)

    #    print("Loaded model from disk")
    #    -------- network --------

    '''

            ##high = np.array(
                [SCREEN_WIDTH // SCALING, SCREEN_HEIGHT // SCALING])  # related to the 2-nd method below
            ##low = np.zeros(2)

            one_player_location_space = spaces.box(low, high, dtype=np.int16)
            # high = np.array([SCREEN_WIDTH//SCALING, SCREEN_HEIGHT//SCALING]*FooEnv.number_of_players)
            # low = np.zeros(2 * FooEnv.number_of_players)
            # all_players_location_space = spaces.box(low, high, dtype=np.int16)

            # 0,1 1d array in size SCREEN_WIDTH/t X SCREEN_HEIGHT/t
            ## game_board_space = spaces.multi_binary(SCREEN_WIDTH//SCALING * SCREEN_HEIGHT//SCALING)
            # 0,1 1d array in size SCREEN_WIDTH/t X SCREEN_HEIGHT/t box of all players with dtype=np.int16
            ## self.observation_space = spaces.tuple(one_player_location_space, game_board_space)


    # 0,1 1d array in size SCREEN_WIDTH/t X SCREEN_HEIGHT/t for some t, box of player with dtype=np.int16
    # related to the above method: self.observation_space = spaces.tuple(one_player_location_space, game_board_space)

    # 0,1 1d array in size 2*SCREEN_WIDTH/t X 2*SCREEN_HEIGHT/t player allways in center
    '''

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        to_return = None

        actions_to_send = [(self.players[0], action)]

        state_maker = GameStateMaker(self.game)
        for i in range(1, NUMBER_OF_PLAYERS):
            p = self.players[i]
            prediction = self.net.predict(state_maker.get_state(show_board=False, player=p))
            # print("\n --------------- \n prediction \n -----------------")
            net_action = np.argmax(prediction)
            actions_to_send.append((p, net_action))

        for i in range(5):
            to_return = self.one_step(actions_to_send)
            if to_return[2]:
                to_return[0] = self.get_state(show_board=SHOW_BOARD)
                return to_return
        to_return[0] = self.get_state(show_board=SHOW_BOARD)
        # print("to_return[0].shape: ",to_return[0].shape)
        return to_return[0], to_return[1], to_return[2], to_return[3]

    def one_step(self, actions):  # actions = the output of each network, the actions they want to do

        # resized_game_board = skimage.measure.block_reduce(self.game.game_board, (SCREEN_RESIZE_FACTOR,SCREEN_RESIZE_FACTOR), np.max)
        # resized_game_board = np.array([[1 if x > 0 else 0 for x in row] for row in resized_game_board])

        # print(prediction, actions[1])
        # print(" --------------- \n prediction \n ----------------- \n")
        # prediction = self.net.predict(state_maker.get_state(show_board=False, player=self.players[2]))
        # # print("\n --------------- \n prediction \n -----------------")
        # net_action_p3 = np.argmax(prediction)

        self.num_of_steps += 1

        # get action from a random trained network
        # for i in range(1, AchtungEnv.number_of_players):
        #     chosen_network = np.random.choice(AchtungEnv.networks)
        #     pred = chosen_network.predict(self.get_state(id=i + 1))
        #     max = (pred[0], 0)
        #     for i in range(len(pred)):
        #         if pred[i] > max[0]:
        #             max = (pred[i], i)
        #     action_from_trained_network = max[0] - 1
        #     actions_to_send.append((self.players[i], action_from_trained_network))

        # do move
        board_not_processed, players, game_over = self.game.step(actions)
        reward = self.get_reward(game_over, players)
        return [{}, reward, bool(game_over), {}]

    def get_state(self, show_board=False):
        resized_game_board = self.preprocess_board(self.game.game_board)
        # self.observation_space[:,:,1] = self.observation_space[:,:,0]
        # self.observation_space[:,:,0] = resized_game_board
        obs = resized_game_board
        # self.show_board(resized_game_board, show_board)
        # curr_place = (self.players[id-1].x, self.players[id-1].y)
        # if self.num_of_steps%100 == 0:
        #     plt.subplot(221)
        #     plt.imshow(self.game.game_board)
        #     # self.axis[1].imshow(cv2.resize(self.game.game_board, (self.game.game_board.shape[1] // 2,self.game.game_board.shape[0] // 2)))
        #     # plt.subplot(222)
        #     # plt.imshow(cv2.resize(self.game.game_board, (40, 40) ))
        #     plt.subplot(223)
        #     plt.imshow(self.observation_space[:,:,0])
        #     plt.subplot(224)
        #     plt.imshow(self.observation_space[:,:,1])
        #     plt.show()
        # return self.observation_space

        # TODO: TORUN
        # ---------------------------------------
        # curr_player = self.players[0]
        # theta = curr_player.theta
        # obs = np.array(list(resized_game_board) +  [np.sin(theta), np.cos(theta)])
        # ---------------------------------------
        return obs

    def preprocess_board(self, board):
        new_board = self.cut_board3_speedy(board, self.players[0])

        # self.add_all_players_to_cutted_board(new_board, ZOOM)
        # curr_player = self.players[0]
        # new_board = rotate(new_board, curr_player.theta + np.pi) #this function rotates the board to match the player
        # if self.num_of_steps % 100 == 0:
        #     plt.imshow(new_board)
        #     plt.show()
        if self.num_of_steps % 10 == 0 and SHOW_BOARD:
            plt.subplot(221)
            plt.imshow(self.game.game_board)
            # self.axis[1].imshow(cv2.resize(self.game.game_board, (self.game.game_board.shape[1] // 2,self.game.game_board.shape[0] // 2)))
            plt.subplot(222)
            plt.imshow(new_board)
            plt.subplot(223)
            plt.imshow(self.smear_board_v2(new_board))
            # plt.subplot(224)
            # plt.imshow(cv2.resize(cv2.resize(new_board, (FINAL_SIZE, FINAL_SIZE)), (15,15) ))
            plt.show()

        new_board = self.smear_board_v2(new_board)
        # new_board = cv2.resize(new_board, (FINAL_SIZE, FINAL_SIZE))

        # prints boards side by side

        # TODO: TORUN
        # ---------------------------------------
        # new_board = new_board.flatten()
        # ---------------------------------------

        # new_board[int(len(board[0])//SCALING * (i // SCALING) + j // SCALING)] = -1
        self.observation_space = new_board
        return self.observation_space

        # TODO: TORUN
        # return new_board

    def show_board(self, board, show_board):
        if not show_board:
            return

        print("--------------- Game Board ---------------" + str(self.players[0]) + "\nturn:" + str(
            self.num_of_steps) + "\n")
        for i in range(self.observation_board_height):
            for j in range(self.observation_board_width):
                if board[self.observation_board_width * i + j] == 2:
                    print(Fore.BLUE + str(board[self.observation_board_width * i + j]) + " ", end="")
                elif board[self.observation_board_width * i + j]:
                    print(Fore.GREEN + str(board[self.observation_board_width * i + j]) + " ", end="")
                else:
                    print(Fore.BLACK + str(board[self.observation_board_width * i + j]) + " ", end="")
            print()

        print("----------- End Of Game Board -----------")

    def in_board_range(self, x, y):
        return 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT

    # def set_val_on_new_board(self, x, x, new_board, val):
    #     row = (SCREEN_WIDTH * ZOOM + rel_x)
    #     col = (SCREEN_HEIGHT * ZOOM + rel_y)
    #     index = int(row_jump + col)
    #     new_board[index] = val

    def add_all_players_to_cutted_board(self, new_board, ZOOM):
        curr_player = self.players[0]
        for p in self.players:
            if abs(p.x - curr_player.x) > SCREEN_WIDTH // (ZOOM * 2) or abs(p.y - curr_player.y) > SCREEN_HEIGHT // (
                    ZOOM * 2):
                continue
            self.draw_circle_on_matrix(new_board,
                                       SCREEN_WIDTH // (ZOOM * 2) + int(p.x) - int(curr_player.x),
                                       SCREEN_HEIGHT // (ZOOM * 2) + int(p.y) - int(curr_player.y),
                                       1)
        # self.draw_circle_on_matrix(new_board, SCREEN_WIDTH // (ZOOM * 2), SCREEN_HEIGHT // (ZOOM * 2), 2)

    def draw_circle_on_matrix(self, new_board, x, y, val):

        for i in range(-RADIUS, RADIUS + 1):
            if x + i < 0 or x + i >= len(new_board):
                continue
            for j in range(-RADIUS, RADIUS + 1):
                if y + j >= len(new_board[i]) or y + j < 0:
                    continue
                new_board[x + i][y + j] = val

    def cut_board1(self, board, player):
        """
        1st gen
        """
        p = player
        new_board = np.zeros((SCREEN_WIDTH // ZOOM, SCREEN_HEIGHT // ZOOM))
        for x in range(SCREEN_WIDTH // ZOOM):
            for y in range(SCREEN_HEIGHT // ZOOM):
                if not self.in_board_range(x - SCREEN_WIDTH // (ZOOM * 2) + int(p.x),
                                           y - SCREEN_HEIGHT // (ZOOM * 2) + int(p.y)):
                    self.draw_circle_on_matrix(new_board, x, y, -1)
                elif board[x - SCREEN_WIDTH // (ZOOM * 2) + int(p.x)][y - SCREEN_HEIGHT // (ZOOM * 2) + int(p.y)]:
                    self.draw_circle_on_matrix(new_board, x, y, -1)
        # for other_player in self.players:
        #     if abs(other_player.x - p.x) > SCREEN_WIDTH // (ZOOM * 2) or abs(other_player.y - p.y) > SCREEN_HEIGHT // (
        #             ZOOM * 2):
        #         continue
        #     self.draw_circle_on_matrix(new_board, SCREEN_WIDTH // (ZOOM * 2) + int(other_player.x) - int(p.x),
        #                                SCREEN_HEIGHT // (ZOOM * 2) + int(other_player.y) - int(p.y), 1)
        self.draw_circle_on_matrix(new_board, SCREEN_WIDTH // (ZOOM * 2), SCREEN_HEIGHT // (ZOOM * 2), 2)
        return new_board

    # def cut_board(self, board):
    #     p = self.players[0]
    #     new_board = np.zeros((SCREEN_WIDTH // ZOOM, SCREEN_HEIGHT // ZOOM)  )
    #     for x in range(SCREEN_WIDTH // ZOOM):
    #         x_temp = x - SCREEN_WIDTH // (ZOOM * 2) + int(p.x)
    #         if 0 > x_temp or x_temp > SCREEN_WIDTH:
    #             continue
    #         for y in range(SCREEN_HEIGHT // ZOOM):
    #             y_temp = y - SCREEN_HEIGHT // (ZOOM * 2) + int(p.y)
    #             if (0 <= x_temp < SCREEN_WIDTH) and (y_temp == 0 or y_temp == SCREEN_HEIGHT):
    #                 self.draw_circle_on_matrix(new_board, x, y, -1)
    #             if (x_temp == 0 or x_temp == SCREEN_WIDTH) and (0 <= y_temp < SCREEN_HEIGHT):
    #                 self.draw_circle_on_matrix(new_board, x, y, -1)
    #
    #             if self.in_board_range(x_temp, y_temp) and board[x_temp][y_temp]:
    #                 self.draw_circle_on_matrix(new_board, x, y, -1)

    def cut_board2(self, board, player):
        """
        2nd gen - fences
        """
        p = player
        new_board = np.zeros((SCREEN_WIDTH // ZOOM, SCREEN_HEIGHT // ZOOM))
        for x in range(SCREEN_WIDTH // ZOOM):
            x_temp = x - SCREEN_WIDTH // (ZOOM * 2) + int(p.x)
            if 0 > x_temp or x_temp > SCREEN_WIDTH:
                continue
            for y in range(SCREEN_HEIGHT // ZOOM):
                y_temp = y - SCREEN_HEIGHT // (ZOOM * 2) + int(p.y)
                if (0 <= x_temp < SCREEN_WIDTH) and (y_temp == 0 or y_temp == SCREEN_HEIGHT):
                    self.draw_circle_on_matrix(new_board, x, y, -1)
                if (x_temp == 0 or x_temp == SCREEN_WIDTH) and (0 <= y_temp < SCREEN_HEIGHT):
                    self.draw_circle_on_matrix(new_board, x, y, -1)

                if self.in_board_range(x_temp, y_temp) and board[x_temp][y_temp]:
                    self.draw_circle_on_matrix(new_board, x, y, -1)

        # for other_player in self.players:
        #     if abs(other_player.x - p.x) > SCREEN_WIDTH // (ZOOM * 2) or abs(other_player.y - p.y) > SCREEN_HEIGHT // (
        #             ZOOM * 2):
        #         continue
        #     self.draw_circle_on_matrix(new_board, SCREEN_WIDTH // (ZOOM * 2) + int(other_player.x) - int(p.x),
        #                                SCREEN_HEIGHT // (ZOOM * 2) + int(other_player.y) - int(p.y), 1)
        # self.draw_circle_on_matrix(new_board, SCREEN_WIDTH // (ZOOM * 2), SCREEN_HEIGHT // (ZOOM * 2), 2)
        return new_board

    def rotate_with_scipy(self, matrix, angle):
        angle_in_deg = 180 - angle * 360 / (2 * np.pi)
        to_return = ndimage.rotate(matrix, angle_in_deg, reshape=False)

        for i in range(len(to_return)):
            for j in range(len(to_return[i])):
                val = to_return[i][j]
                if -1.5 < val < -0.5:
                    to_return[i][j] = -1
                else:
                    to_return[i][j] = 0
        return to_return

    def cut_board3_first_cut(self, board, player):
        p = player
        new_board = np.zeros((int(1.5 * SCREEN_WIDTH // ZOOM + 10), int(1.5 * SCREEN_HEIGHT // ZOOM + 10)))
        for x in range(int(1.5 * SCREEN_WIDTH // ZOOM + 10)):
            x_temp = x - int(1.5 * SCREEN_WIDTH // (2 * ZOOM) + 10) + int(p.x)
            if 0 > x_temp or x_temp > SCREEN_WIDTH:
                continue
            for y in range(int(1.5 * SCREEN_HEIGHT // ZOOM + 10)):
                y_temp = y - int(1.5 * SCREEN_HEIGHT // (ZOOM * 2) + 10) + int(p.y)
                if (0 <= x_temp < SCREEN_WIDTH) and (y_temp == 0 or y_temp == SCREEN_HEIGHT):
                    self.draw_circle_on_matrix(new_board, x, y, -1)
                if (x_temp == 0 or x_temp == SCREEN_WIDTH) and (0 <= y_temp < SCREEN_HEIGHT):
                    self.draw_circle_on_matrix(new_board, x, y, -1)

                if self.in_board_range(x_temp, y_temp) and board[x_temp][y_temp]:
                    self.draw_circle_on_matrix(new_board, x, y, -1)
        return new_board

    def cut_board3(self, board, player):
        """
        3rd gen - rotate
        """

        new_board = self.cut_board3_first_cut(board, player);
        centered_player = (int((1.5 * SCREEN_WIDTH // ZOOM + 10) // 2), int(1.5 * SCREEN_HEIGHT // ZOOM + 10) // 2)
        new_board = self.rotate_with_scipy(new_board, player.theta)
        center_X, center_y = len(new_board) // 2, len(new_board[0]) // 2
        new_board = new_board[
                    center_X - (SCREEN_WIDTH // (2 * ZOOM)): center_X + (SCREEN_WIDTH // (2 * ZOOM)),
                    center_y - (SCREEN_HEIGHT // (2 * ZOOM)): center_y + (SCREEN_HEIGHT // (2 * ZOOM))
                    ]
        return new_board

    def cut_board3_speedy(self, board, player):
        """
        3rd gen - rotate (speedy version)
        """

        new_board = self.cut_board2(board, player)
        new_board = self.smear_board_v2(new_board, size=int(FINAL_SIZE * 2))
        new_board = self.rotate_with_scipy(new_board, player.theta)

        return new_board

    def smear_board_v1(self, new_board):
        smeard_board = np.zeros((len(new_board) // SCALING + 1, len(new_board[0]) // SCALING + 1))
        for x in range(len(new_board)):
            for y in range(len(new_board[0])):
                # fix1 = 0
                # fix2 = 0
                # if x//SCALING == len(smear_board):
                #     fix1 = 1
                # if y//SCALING == len(smear_board[x//SCALING - fix]):
                #     fix2 = 1
                if new_board[x][y]:
                    if smeard_board[x // SCALING][y // SCALING] == 0 or new_board[x][y] > 0:
                        smeard_board[x // SCALING][y // SCALING] = new_board[x][y]

        return smeard_board

    def get_smeared_board_one_val(self, new_board, x, y, size=FINAL_SIZE):
        jump_size = len(new_board) // size
        to_return = 0
        for tmp_x in range(x * jump_size, (x + 1) * jump_size):
            for tmp_y in range(y * jump_size, (y + 1) * jump_size):
                if new_board[tmp_x][tmp_y]:
                    if new_board[x][y] > 0:
                        return new_board[tmp_x][tmp_y]
                    to_return = new_board[tmp_x][tmp_y]

        return to_return

    def smear_board_v2(self, new_board, size=FINAL_SIZE):
        smeared_board = np.zeros((size, size))

        for x in range(size):
            for y in range(size):
                smeared_board[x][y] = self.get_smeared_board_one_val(new_board, x, y, size=size)

        return smeared_board

    def get_reward(self, game_over, players):
        """
        Reward with many players in the game
        if not game_over and self.players[0] in players:
            return -len(players)
        elif game_over and self.players[0] in players:
            return WIN_REWARD
        """
        if self.game.get_player_by_id(1) is not None:
            return 1.0
        else:
            return -100.0

    def reset(self):
        self.game = AchtungGame(AchtungEnv.number_of_players)
        self.players = [self.game.get_player_by_id(id + 1) for id in range(AchtungEnv.number_of_players)]
        self.seed()
        self.num_of_steps = 0
        self.game_recorder = []
        plt.close('all')

        # ------ pygame ------
        # self.screen.fill((0, 0, 0))

        return self.get_state(show_board=SHOW_BOARD)

    def render(self, mode='human', close=False):
        pass
