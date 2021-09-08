import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from colorama import Fore
from scipy import ndimage
from keras.models import model_from_json
from gym_achtung.envs.ahtungGame import AchtungGame
from gym_achtung.envs.consts import *
from gym_achtung.envs.state_maker import GameStateMaker


class AchtungEnv(gym.Env):
    # networks = []
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players=1):
        # ------ pygame ------

        # simulation:
        self.reward = 0
        self.number_players = num_players
        self.game = AchtungGame(self.number_players)
        self.players = [self.game.get_player_by_id(player_id + 1) for player_id in range(self.number_players)]
        self.action_space = spaces.Discrete(3)

        self.observation_board_width = FINAL_SIZE
        self.observation_board_height = FINAL_SIZE

        self.observation_space = np.zeros((self.observation_board_height, self.observation_board_width))
        self.num_of_steps = 0

        self.seed()

        self._board_width, self._board_height = self.game.game_board.shape

        #    -------- network --------
        exp3_model = r'F:\Projects\Achtung Die Kurve\Dev\models\Good Models\trainedmodel_exp_rotate.json'
        exp3_weights = r'F:\Projects\Achtung Die Kurve\Dev\models\Good Models\model_weights_exp_rotate.h5'

        json_file = open(exp3_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.net = model_from_json(loaded_model_json)
        self.net.load_weights(exp3_weights)
    #    -------- network --------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        to_return = None

        actions_to_send = [(self.players[0], action)]

        state_maker = GameStateMaker(self.game)
        for i in range(1, self.number_players):
            p = self.players[i]
            prediction = self.net.predict(state_maker.get_state(show_board=False, player=p))
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
        self.num_of_steps += 1

        # do move
        board_not_processed, players, game_over = self.game.step(actions)
        reward = self.get_reward(game_over, players)
        return [{}, reward, bool(game_over), {}]

    def get_state(self, show_board=False):
        resized_game_board = self.preprocess_board(self.game.game_board)
        obs = resized_game_board
        return obs

    def preprocess_board(self, board):
        new_board = self.cut_board3_speedy(board, self.players[0])

        if self.num_of_steps % 10 == 0 and SHOW_BOARD:
            plt.subplot(221)
            plt.imshow(self.game.game_board)
            plt.subplot(222)
            plt.imshow(new_board)
            plt.subplot(223)
            plt.imshow(self.smear_board_v2(new_board))
            plt.show()

        new_board = self.smear_board_v2(new_board)

        self.observation_space = new_board
        return self.observation_space

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
        return 0 <= x < self._board_width and 0 <= y < self._board_height

    def add_all_players_to_cutted_board(self, new_board, ZOOM):
        curr_player = self.players[0]
        for p in self.players:
            if abs(p.x - curr_player.x) > self._board_width // (ZOOM * 2) or abs(p.y - curr_player.y) > self._board_height // (
                    ZOOM * 2):
                continue
            self.draw_circle_on_matrix(new_board,
                                       self._board_width // (ZOOM * 2) + int(p.x) - int(curr_player.x),
                                       self._board_height // (ZOOM * 2) + int(p.y) - int(curr_player.y),
                                       1)

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
        new_board = np.zeros((self._board_width // ZOOM, self._board_height // ZOOM))
        for x in range(self._board_width // ZOOM):
            for y in range(self._board_height // ZOOM):
                if not self.in_board_range(x - self._board_width // (ZOOM * 2) + int(p.x),
                                           y - self._board_height // (ZOOM * 2) + int(p.y)):
                    self.draw_circle_on_matrix(new_board, x, y, -1)
                elif board[x - self._board_width // (ZOOM * 2) + int(p.x)][y - self._board_height // (ZOOM * 2) + int(p.y)]:
                    self.draw_circle_on_matrix(new_board, x, y, -1)
        self.draw_circle_on_matrix(new_board, self._board_width // (ZOOM * 2), self._board_height // (ZOOM * 2), 2)
        return new_board

    def cut_board2(self, board, player):
        """
        2nd gen - fences
        """
        p = player
        new_board = np.zeros((self._board_width // ZOOM, self._board_height // ZOOM))
        for x in range(self._board_width // ZOOM):
            x_temp = x - self._board_width // (ZOOM * 2) + int(p.x)
            if 0 > x_temp or x_temp > self._board_width:
                continue
            for y in range(self._board_height // ZOOM):
                y_temp = y - self._board_height // (ZOOM * 2) + int(p.y)
                if (0 <= x_temp < self._board_width) and (y_temp == 0 or y_temp == self._board_height):
                    self.draw_circle_on_matrix(new_board, x, y, -1)
                if (x_temp == 0 or x_temp == self._board_width) and (0 <= y_temp < self._board_height):
                    self.draw_circle_on_matrix(new_board, x, y, -1)

                if self.in_board_range(x_temp, y_temp) and board[x_temp][y_temp]:
                    self.draw_circle_on_matrix(new_board, x, y, -1)

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
        new_board = np.zeros((int(1.5 * self._board_width // ZOOM + 10), int(1.5 * self._board_height // ZOOM + 10)))
        for x in range(int(1.5 * self._board_width // ZOOM + 10)):
            x_temp = x - int(1.5 * self._board_width // (2 * ZOOM) + 10) + int(p.x)
            if 0 > x_temp or x_temp > self._board_width:
                continue
            for y in range(int(1.5 * self._board_height // ZOOM + 10)):
                y_temp = y - int(1.5 * self._board_height // (ZOOM * 2) + 10) + int(p.y)
                if (0 <= x_temp < self._board_width) and (y_temp == 0 or y_temp == self._board_height):
                    self.draw_circle_on_matrix(new_board, x, y, -1)
                if (x_temp == 0 or x_temp == self._board_width) and (0 <= y_temp < self._board_height):
                    self.draw_circle_on_matrix(new_board, x, y, -1)

                if self.in_board_range(x_temp, y_temp) and board[x_temp][y_temp]:
                    self.draw_circle_on_matrix(new_board, x, y, -1)
        return new_board

    def cut_board3(self, board, player):
        """
        3rd gen - rotate
        """

        new_board = self.cut_board3_first_cut(board, player)
        centered_player = (int((1.5 * self._board_width // ZOOM + 10) // 2), int(1.5 * self._board_height // ZOOM + 10) // 2)
        new_board = self.rotate_with_scipy(new_board, player.theta)
        center_X, center_y = len(new_board) // 2, len(new_board[0]) // 2
        new_board = new_board[
                    center_X - (self._board_width // (2 * ZOOM)): center_X + (self._board_width // (2 * ZOOM)),
                    center_y - (self._board_height // (2 * ZOOM)): center_y + (self._board_height // (2 * ZOOM))
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
        if self.game.get_player_by_id(1) is not None:
            return 1.0
        else:
            return -100.0

    def reset(self):
        self.game = AchtungGame(self.number_players)
        self.players = [self.game.get_player_by_id(player_id + 1) for player_id in range(self.number_players)]
        self.seed()
        self.num_of_steps = 0
        plt.close('all')

        return self.get_state(show_board=SHOW_BOARD)

    def render(self, mode='human', close=False):
        pass
