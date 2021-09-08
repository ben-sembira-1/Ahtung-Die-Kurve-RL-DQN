# --------- Imports ---------
from gym_achtung.envs.consts import *
import numpy as np
import time
# from gym_achtung.envs.temp_queue import TempQueue
from collections import deque
from keras.models import Sequential, model_from_json
import cv2
from scipy import ndimage

# ------ End Of Imports -----

# ------ Constants ------
MODEL_FILE = r'F:\Projects\Achtung Die Kurve\Dev\models\Good Models\trainedmodel_exp_rotate.json'
MODEL_WEIGHTS_FILE = r'F:\Projects\Achtung Die Kurve\Dev\models\Good Models\model_weights_exp_rotate.h5'

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
        obs = resized_game_board
        self.show_board(resized_game_board, show_board)
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
        new_board = self.cut_board3_speedy(board, player)
        new_board = self.smear_board_v2(new_board)

        # TODO: TORUN

        return new_board

    def show_board(self, board, show_board):
        pass

    def in_board_range(self, x, y):
        return 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT

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

        return new_board


# ------------------------------------------------------------------------------
# ----------------------------------        ------------------------------------
# --------------------------------   Player   ----------------------------------
# ----------------------------------        ------------------------------------
# ------------------------------------------------------------------------------


class Player:

    def __init__(self, other_players, player_id):
        self.id = player_id
        # 0 for x, 1 for y
        self.x, self.y = self._generate_rnd_coor(other_players)
        self.previous_points = deque(maxlen=DEFAULT_MAX_QUEUE_SIZE)
        self.previous_points.append((int(self.x), int(self.y)))
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
        if not self.is_gapping and self.curr_turn - self.turn_last_gap_started > MAKE_GAP_PERIOD:
            if np.random.random() < GAP_EPSILON:
                self.is_gapping = True
                self.turn_last_gap_started = self.curr_turn

        if self.is_gapping and self.curr_turn - self.turn_last_gap_started >= MAKE_GAP_DURATION:
            self.is_gapping = False

        if not check and not self.is_gapping:
            self.previous_points.append((int(self.x), int(self.y)))

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
                    self.previous_points.append((int(self.x), int(self.y)))

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

    def check_on_board(self, game_board, x, y):
        for i in range(-1 * int(CIRCLE_SIZE * 1.5), int(CIRCLE_SIZE * 1.5) + 1):
            if i == 0:
                continue
            if 0 <= int(x) + i < SCREEN_WIDTH and 0 <= int(y) + i < SCREEN_HEIGHT:
                if game_board[int(x) + i][int(y)] != 0 and not self.previous_points.count((int(x) + i, int(y))):
                    return False
                if game_board[int(x)][int(y) + i] != 0 and not self.previous_points.count((int(x), int(y) + i)):
                    return False
        bool = game_board[int(x)][int(y)] == 0 or self.previous_points.count((int(x), int(y)))
        return bool

    def in_bounds(self, x_dest, y_dest):
        return 0 < x_dest < SCREEN_WIDTH and 0 < y_dest < SCREEN_HEIGHT

# ------------------------------------------------------------------------------
# -------------------------------              ---------------------------------
# -----------------------------   AchtungGame    ------------------------------
# -------------------------------              ---------------------------------
# ------------------------------------------------------------------------------

class AchtungGame:
    num_of_turns = 0

    def __init__(self, number_of_players=NUMBER_OF_PLAYERS):
        self.game_board = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.players = []
        for i in range(number_of_players):
            new_player = Player(self.players, i + 1)
            self.players.append(new_player)
        self.game_over = False

    def __str__(self):
        str = ""
        for p in self.players:
            str += p.__str__()
            str += "\n"
        return str

    def get_player_by_id(self, id):
        for p in self.players:
            if p.id == id:
                return p
        return None

    def step(self, actions):
        AchtungGame.num_of_turns += 1

        game_over = False
        actions = [action for action in actions if action[0] in self.players]

        for player, input in actions:
            is_player_still_alive = player.update(self.game_board, FROM_INPUT_TO_THETA_CHANGE[input])

            if not is_player_still_alive:
                self.players.remove(player)

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


        # ------ pygame ------
        pygame.init()
        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        self.screen.fill((0, 0, 0))
        # ---- pygame end ----

        # -------- network --------
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

    # -2: game-over, -1: go-left, 0: do-nothing, 1: go-right
    def get_input_from_user(self):
        actions = {}
        for i in range(len(self.game.players)):
            actions[i + 1] = INPUT["do-nothing"]

        state_maker = GameStateMaker(self.game)
        for i in range(0, NUMBER_OF_PLAYERS):
            player = self.game.get_player_by_id(i + 1)
            if not player:
                continue

            net = self.net1

            prediction = net.predict(state_maker.get_state(player=player))

            # print("\n --------------- \n prediction \n -----------------")
            actions[i + 1] = np.argmax(prediction)
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

        for p in self.game.players:
            if p.previous_points:
                pygame.draw.circle(
                    self.screen,
                    COLORS[p.id],
                    (int(p.previous_points[-1][0]), int(p.previous_points[-1][1])),
                    CIRCLE_SIZE
                )


if __name__ == "__main__":
    runner = AchtungGameRunner(NUMBER_OF_PLAYERS)
    runner.run_game()

# TODO: fix gaps to be based on turns and not time
# TODO: first priority - improve existing player
# TODO: one method:
