# --------- Imports ---------
from gym_achtung.envs.consts import *
import numpy as np
import time
from gym_achtung.envs.helper import TmpQueue
from keras.models import Sequential, model_from_json
import cv2
from scipy import ndimage


# ------ End Of Imports -----

# ------ Constants ------
MODEL_FILE = r'F:\Projects\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\trainedmodel_exp_rotate.json'
MODEL_WEIGHTS_FILE = r'F:\Projects\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\model_weights_exp_rotate.h5'

class GameStateMaker(object):
    """docstring for GameStateMaker."""

    def __init__(self, game):
        super(GameStateMaker, self).__init__()
        self.game = game
        self.players = [self.game.get_player_by_id(id + 1) for id in range(1)]  # TODO: 1 is for number of players

    def get_state(self, show_board=False, player=None):
        if not player:
            player = self.players[0]
        resized_game_board = self.preprocess_board(self.game.game_board, player)
        # self.observation_space[:,:,1] = self.observation_space[:,:,0]
        # self.observation_space[:,:,0] = resized_game_board
        obs = resized_game_board
        self.show_board(resized_game_board, show_board)
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
        return np.array([[obs]])

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

    def cut_board3_speedy(self, board, player):
        """
        3rd gen - rotate (speedy version)
        """

        new_board = self.cut_board2(board, player)
        new_board = self.smear_board_v2(new_board, size=int(FINAL_SIZE * 2))
        new_board = self.rotate_with_scipy(new_board, player.theta)

        return new_board

    def smear_board_v2(self, new_board, size=FINAL_SIZE):
        smeared_board = np.zeros((size, size))

        for x in range(size):
            for y in range(size):
                smeared_board[x][y] = self.get_smeared_board_one_val(new_board, x, y, size=size)

        return smeared_board

    def preprocess_board(self, board, player):
        # new_board = self.cut_board(board, player)
        new_board = self.cut_board3_speedy(board, player)

        # self.add_all_players_to_cutted_board(new_board, ZOOM)
        new_board = self.smear_board_v2(new_board)
        # curr_player = self.players[0]
        # new_board = rotate(new_board, curr_player.theta + np.pi) #this function rotates the board to match the player
        # if self.num_of_steps % 100 == 0:
        #     plt.imshow(new_board)
        #     plt.show()
        # if self.num_of_steps%50 == 0:
        #     plt.subplot(221)
        #     plt.imshow(self.game.game_board)
        #     # self.axis[1].imshow(cv2.resize(self.game.game_board, (self.game.game_board.shape[1] // 2,self.game.game_board.shape[0] // 2)))
        #     plt.subplot(222)
        #     plt.imshow(new_board)
        #     plt.subplot(223)
        #     plt.imshow(cv2.resize(new_board, (FINAL_SIZE, FINAL_SIZE) ))
        #     plt.subplot(224)
        #     plt.imshow(cv2.resize(cv2.resize(new_board, (FINAL_SIZE, FINAL_SIZE)), (15,15) ))
        #     plt.show()
        # new_board = cv2.resize(new_board, (FINAL_SIZE, FINAL_SIZE))

        # prints boards side by side

        # TODO: TORUN
        # ---------------------------------------
        # new_board = new_board.flatten()
        # ---------------------------------------

        # new_board[int(len(board[0])//SCALING * (i // SCALING) + j // SCALING)] = -1

        return new_board
        # self.observation_space = new_board
        # return self.observation_space

        # TODO: TORUN
        # return new_board

    def show_board(self, board, show_board):
        pass

    #     if not show_board:
    #         return
    #
    #     print("--------------- Game Board ---------------" + str(self.players[0]) + "\nturn:" + str(self.num_of_steps) + "\n")
    #     for i in range(self.observation_board_height):
    #         for j in range(self.observation_board_width):
    #             if board[self.observation_board_width * i + j] == 2:
    #                 print(Fore.BLUE + str(board[self.observation_board_width * i + j]) + " ",end = "")
    #             elif board[self.observation_board_width * i + j]:
    #                 print(Fore.GREEN + str(board[self.observation_board_width * i + j]) + " ",end = "")
    #             else:
    #                 print(Fore.BLACK + str(board[self.observation_board_width * i + j]) + " ", end = "")
    #         print()
    #
    #     print("----------- End Of Game Board -----------")

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
        self.draw_circle_on_matrix(new_board, SCREEN_WIDTH // (ZOOM * 2), SCREEN_HEIGHT // (ZOOM * 2), 2)

    def draw_circle_on_matrix(self, new_board, x, y, val):

        for i in range(-RADIUS, RADIUS + 1):
            if x + i < 0 or x + i >= len(new_board):
                continue
            for j in range(-RADIUS, RADIUS + 1):
                if y + j >= len(new_board[i]) or y + j < 0:
                    continue
                new_board[x + i][y + j] = val

    # def cut_board(self, board, player):
    #     p = player
    #     new_board = np.zeros((SCREEN_WIDTH // ZOOM, SCREEN_HEIGHT // ZOOM))
    #     for x in range(SCREEN_WIDTH // ZOOM):
    #         for y in range(SCREEN_HEIGHT // ZOOM):
    #             if not self.in_board_range(x - SCREEN_WIDTH // (ZOOM * 2) + int(p.x),
    #                                        y - SCREEN_HEIGHT // (ZOOM * 2) + int(p.y)):
    #                 self.draw_circle_on_matrix(new_board, x, y, -1)
    #             elif board[x - SCREEN_WIDTH // (ZOOM * 2) + int(p.x)][y - SCREEN_HEIGHT // (ZOOM * 2) + int(p.y)]:
    #                 self.draw_circle_on_matrix(new_board, x, y, -1)
    #     for other_player in self.players:
    #         if abs(other_player.x - p.x) > SCREEN_WIDTH // (ZOOM * 2) or abs(other_player.y - p.y) > SCREEN_HEIGHT // (
    #                 ZOOM * 2):
    #             continue
    #         self.draw_circle_on_matrix(new_board, SCREEN_WIDTH // (ZOOM * 2) + int(other_player.x) - int(p.x),
    #                                    SCREEN_HEIGHT // (ZOOM * 2) + int(other_player.y) - int(p.y), 1)
    #     self.draw_circle_on_matrix(new_board, SCREEN_WIDTH // (ZOOM * 2), SCREEN_HEIGHT // (ZOOM * 2), 2)
    #     return new_board
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

    def cut_board(self, board, player):
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


# ------------------------------------------------------------------------------
# ----------------------------------        ------------------------------------
# --------------------------------   Player   ----------------------------------
# ----------------------------------        ------------------------------------
# ------------------------------------------------------------------------------


class Player:

    def __init__(self, other_players, id):
        self.id = id
        # 0 for x, 1 for y
        self.x, self.y = self._generate_rnd_coor(other_players)
        self.previous_points = TmpQueue()
        self.previous_points.push((int(self.x), int(self.y)))
        self.theta = np.random.random() * 2 * np.pi
        self.turn_last_gap_started = 0
        self.curr_turn = 0
        self.is_gapping = False

    def __str__(self):
        return "PLAYER: id-{self.id}, position-({self.x}, {self.y}), theta-{self.theta}".format(self = self)

    def get(self, i):
        if i == 0:
            return self.x
        if i == 1:
            return self.y
        return None

    def _generate_rnd_coor(self, other_players):

        if other_players == []:
            new_point = [np.random.randint( int(SCREEN_WIDTH//4), int(3*SCREEN_WIDTH//4)), np.random.randint(int(SCREEN_HEIGHT//4), int(3*SCREEN_HEIGHT//4))]
            return new_point

        def check_coordinates(point, other_points):
            for other_point in other_points:
                curr_dist = np.linalg.norm([other_point[0] - point[0],other_point[1] - point[1]])
                if curr_dist < MIN_DIST_BETWEEN_SNAKES_START:
                    return False
            return True

        all_coor_taken = [[p.x, p.y] for p in other_players]
        new_point = [np.random.randint( int(SCREEN_WIDTH//4), int(3*SCREEN_WIDTH//4)), np.random.randint(int(SCREEN_HEIGHT//4), int(3*SCREEN_HEIGHT//4))]

        while check_coordinates(new_point, all_coor_taken) is not True:
            new_point = [np.random.randint( int(SCREEN_WIDTH//4), int(3*SCREEN_WIDTH//4)), np.random.randint(int(SCREEN_HEIGHT//4), int(3*SCREEN_HEIGHT//4))]

        return new_point

    def go_on_path(self, game_board, xDestination, yDestination, delta_x, delta_y, check=True):
        # Random gap
        # if self.id == 1:
        #     GAP_EPSILON = 0
        # else:
        #     GAP_EPSILON = 0.002
        if not self.is_gapping and self.curr_turn - self.turn_last_gap_started > MAKE_GAP_PERIOD:
            if np.random.random() < GAP_EPSILON:
                self.is_gapping = True
                self.turn_last_gap_started = self.curr_turn

        if self.is_gapping and self.curr_turn - self.turn_last_gap_started >= MAKE_GAP_DURATION:
            self.is_gapping = False

        if not check and not self.is_gapping:
            self.previous_points.push((int(self.x), int(self.y)))
#aaa
        temp_x, temp_y = self.x, self.y
        x_got_to_place, y_got_to_place = False, False
        while not x_got_to_place or not y_got_to_place:

            # actual step/check-step
            temp_x += delta_x
            temp_y += delta_y
            if not check:
                self.x = temp_x
                self.y = temp_y
                if not self.is_gapping:
                    self.previous_points.push((int(self.x), int(self.y)))

            # runs checker or real player
            if not self.in_bounds(temp_x, temp_y):
                return False
            if not self.is_gapping:
                if check:
                    if not self.check_on_board(game_board, temp_x, temp_y):
                        return False
                else:
                    self.draw_on_board(game_board, temp_x, temp_y)

            if int(temp_x) == xDestination:
                x_got_to_place = True
            if int(temp_y) == yDestination:
                y_got_to_place = True


        return True

    def update(self, game_board, theta_change):
        self.curr_turn += 1
        self.theta += theta_change * DELTA_THETA
        delta_x =  np.cos(self.theta)
        delta_y = np.sin(self.theta)
        xDestination = int(self.x + SPEED*delta_x)
        yDestination = int(self.y + SPEED*delta_y)

        # checks if destination is in the board
        if not self.in_bounds(xDestination, yDestination):
            return False

        # checks if path is ok
        if self.go_on_path(game_board, xDestination, yDestination, delta_x, delta_y, check=True):

            # This is a good move! goes on path
            self.go_on_path(game_board, xDestination, yDestination, delta_x, delta_y, check=False)
            return True

        # path is not ok - player is dead
        return False

    def draw_on_board(self, game_board, x, y):
        game_board[int(x)][int(y)] = self.id
        # if game_board[int(x)][int(y)] == 0:
        # for i in range(-1, 2):
        #     if 0 <= int(x) + i < SCREEN_WIDTH and 0 <= int(y) + i < SCREEN_HEIGHT:
        #         game_board[int(x) + i][int(y)] = self.id
        #         game_board[int(x)][int(y) + i] = self.id

    def check_on_board(self, game_board, x, y):
        for i in range(-1 * int(CIRCLE_SIZE * 1.5), int(CIRCLE_SIZE * 1.5) + 1):
            if i == 0:
                continue
            if 0 <= int(x) + i < SCREEN_WIDTH and 0 <= int(y) + i < SCREEN_HEIGHT:
                if game_board[int(x) + i][int(y)] != 0 and not self.previous_points.contains((int(x) + i, int(y))):
                    return False
                if game_board[int(x)][int(y) + i] != 0 and not self.previous_points.contains((int(x), int(y) + i)):
                    return False
        bool = game_board[int(x)][int(y)] == 0 or self.previous_points.contains((int(x), int(y)))
        return bool

    def in_bounds(self, x_dest, y_dest):
        return 0 < x_dest < SCREEN_WIDTH and 0 < y_dest < SCREEN_HEIGHT

# ------------------------------------------------------------------------------
# -------------------------------              ---------------------------------
# -----------------------------   AchtungGame    ------------------------------
# -------------------------------              ---------------------------------
# ------------------------------------------------------------------------------dddddddaaaaaaaaa

class AchtungGame:
    num_of_turns = 0

    def __init__(self, number_of_players=NUMBER_OF_PLAYERS):
        self.game_board = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.players = []
        for i in range(number_of_players):
            new_player = Player(self.players, i + 1)
            self.players.append(new_player)
        # self.players[0].x = 100
        # self.players[0].y = 50
        # self.players[1].x = 800
        # self.players[1].y = 50
        # self.players[0].theta = 0.25 * np.pi
        # self.players[1].theta = 0.75 * np.pi
        self.game_over = False
        # self.players = np.random.rand(number_of_players,3)

    def __str__(self):
        str = ""
        for p in self.players:
            str += p.__str__()
            str += "\n"
        return str

    # def set_players(number_of_players):
    #     for i in range(number_of_players):
    #         self.players.append(Player(self.players))

    def get_player_by_id(self, id):
        for p in self.players:
            if p.id == id:
                return p
        return None

    def step(self, actions):
        AchtungGame.num_of_turns += 1
        # action = [player, theta_change]
        # theta_change = {-1:turn left, 0:dont change, 1:turn right}

        game_over = False
        # filters only actions for players that are still alive
        actions = [action for action in actions if action[0] in self.players]

        for player, input in actions:
            is_player_still_alive = player.update(self.game_board, FROM_INPUT_TO_THETA_CHANGE[input])

            if not is_player_still_alive:
                self.players.remove(player)

        # print(self.__str__())
        threshold = 0 if NUMBER_OF_PLAYERS == 1 else 1
        if (len(self.players) <= 0):
            game_over = True
        return self.game_board, self.players, game_over



# ------------------------------------------------------------------------------
# --------------------------------            ----------------------------------
# ------------------------------   GameRunner    -------------------------------
# --------------------------------            ----------------------------------
# ------------------------------------------------------------------------------

import pygame
from pygame.locals import KEYDOWN, QUIT


class AchtungGameRunner:

    def __init__(self, num_of_players):
        self.game = AchtungGame(num_of_players)
        self.next_steps = []
        exp3_json = r'trainedmodel_exp3.json'
        exp3_weights = r'model_weights_exp3.h5'


        # ------ pygame ------
        pygame.init()
        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        self.screen.fill((0, 0, 0))
        # ---- pygame end ----

        # -------- network --------
        # -------- network --------
        # v3_model = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\model jasons\trainedmodelv3.json'
        # v3_weights = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\model weights\model_weightsv3.h5'
        # v1_model = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\model jasons\trainedmodel.json'
        # v1_weights = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\model weights\model_weights.h5'
        # exp4_model = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\trainedmodel_exp4.json'
        # exp4_weights = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\model_weights_exp4.h5'
        # exp3_model = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\trainedmodel_exp3.json'
        # exp3_weights = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\model_weights_exp3.h5'
        # exp4_ultra_model = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\trainedmodel_exp4_ultra.json'
        # exp4_ultra_weights = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\model_weights_exp4_ultra.h5'
        # exp4_nogaps_weights = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\model_weights_exp4_nogaps.h5'
        # exp4_nogaps_model = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\trainedmodel_exp4_nogaps.json'
        # fifty_k_model = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\trainedjson\trainedmodel_00_warmup_1000_.json'
        # fifty_k_weights = r'D:\IDF\smop\Achtung Die Kurve\CODE\Ahtung-Die-Kurve-RL-DQN\models\trainedweights\trainedmodel_weights_00_warmup_1000_.h5'

        json_file = open(MODEL_FILE, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.net1 = model_from_json(loaded_model_json)
        self.net1.load_weights(MODEL_WEIGHTS_FILE)
        print("Loaded model from disk")

        json_file = open(MODEL_FILE, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.net2 = model_from_json(loaded_model_json)
        self.net2.load_weights(MODEL_WEIGHTS_FILE)
        print("Loaded model from disk")
        # -------- network --------
        # json_file = open(exp3_json, 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        # self.net = model_from_json(loaded_model_json)
        # self.net.load_weights(exp_rot_weights)
        # print("Loaded model from disk")
        # -------- network --------

    # -2: game-over, -1: go-left, 0: do-nothing, 1: go-right
    def get_input_from_user(self):

        # for event in pygame.event.get():
        #     if event.type == KEYDOWN:
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #             return INPUT["game-over"]
        #
        #
        # key_array = KEY_ARRAY
        # keys_pressed = pygame.key.get_pressed()
        #
        actions = {}
        for i in range(len(self.game.players)):
            actions[i + 1] = INPUT["do-nothing"]
        #
        # for i,keys in enumerate(key_array):
        #     id = i + 1
        #     if keys_pressed[keys[0]]:
        #         actions[id] = INPUT["go-left"]
        #     if keys_pressed[keys[1]]:
        #         if actions[id] == INPUT["do-nothing"]:
        #             actions[id] = INPUT["go-right"]
        #         else:
        #             actions[id] = INPUT["do-nothing"]
        #
        # for keys in key_array:
        #     if keys[0] in actions and keys[1] in actions:
        #         actions.remove(keys[0])
        #         actions.remove(keys[1])

        state_maker = GameStateMaker(self.game)
        for i in range(0, NUMBER_OF_PLAYERS):
            player = self.game.get_player_by_id(i + 1)
            if not player:
                continue

            net = self.net1
            # if i != 0:
            #     net = self.net2
                # size = 20

            prediction = net.predict(state_maker.get_state(player=player))

            # print("\n --------------- \n prediction \n -----------------")
            actions[i + 1] = np.argmax(prediction)
            # print(prediction, actions[1])
            # print(" --------------- \n prediction \n ----------------- \n")

        return actions

    def run_game(self):
        runing = True
        while runing:
            input = self.get_input_from_user()
            if input == INPUT["game-over"]:
                break
            self.next_steps = [[player, input[player.id]] for player in
                               self.game.players]  # assigns each player the operation it has to do now
            for i in range(5):
                game_board, players, game_over = self.game.step(self.next_steps)
                if game_over:
                    print("Game is over")
                    runing = False
                    break

                self.render_game()
                pygame.display.flip()
            # pygame.time.wait(REFRESH_SPEED)

    def render_game(self):
        # pygame.draw.circle(self.screen, (0, 0, 255), (250, 250), 75)
        # for i in range(SCREEN_WIDTH):
        #     for j in range(SCREEN_HEIGHT):
        #         p_id = self.game.game_board[i][j]
        #         if p_id:
        #             pygame.draw.circle(
        #                         self.screen,
        #                         COLORS[p_id],
        #                         (i // 4 , j // 4),
        #                         CIRCLE_SIZE
        #                     )


        for p in self.game.players:

            pygame.draw.circle(
                self.screen,
                COLORS[p.id],
                (int(p.previous_points.top()[0]), int(p.previous_points.top()[1])),
                CIRCLE_SIZE
            )
            # pygame.draw.lines(
            #             self.screen,
            #             COLORS[p.id],
            #             False,d
            #             [(int(p.previous_points.top()[0]) // 4, int(p.previous_points.top()[1]) // 4), (int(p.x)//4 , int(p.y) // 4)],
            #             10
            #         )


if __name__ == "__main__":
    runner = AchtungGameRunner(NUMBER_OF_PLAYERS)
    runner.run_game()

# TODO: fix gaps to be based on turns and not time
# TODO: first priority - improve existing player
# TODO: one method:
