
# --------- Imports ---------
from consts import *
import numpy as np
import time
from helper import TmpHeap
# ------ End Of Imports -----


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
        self.previous_points = TmpHeap()
        self.previous_points.push((int(self.x), int(self.y)))
        self.theta =  np.random.random() * 2*np.pi
        self.time_last_gap_started = time.time()
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
        if not self.is_gapping and (time.time() - (self.time_last_gap_started + MAKE_GAP_DURATION)) > MAKE_GAP_PERIOD:
            if np.random.random() < GAP_EPSILON:
                self.is_gapping = True
                self.time_last_gap_started = time.time()

        if self.is_gapping and time.time() - self.time_last_gap_started >= MAKE_GAP_DURATION:
            self.is_gapping = False

        if not check and not self.is_gapping:
            self.previous_points.push((int(self.x), int(self.y)))

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
        for i in range(-1 * CIRCLE_SIZE * 8, CIRCLE_SIZE * 8 + 1):
            if i == 0:
                continue
            if 0 <= int(x) + i < SCREEN_WIDTH and 0 <= int(y) + i < SCREEN_HEIGHT:
                if game_board[int(x) + i][int(y)] != 0 and not self.previous_points.contains((int(x) + i, int(y))):
                    print(1)
                    return False
                if game_board[int(x)][int(y) + i] != 0 and not self.previous_points.contains((int(x), int(y) + i)):
                    print(2)
                    return False
        bool = game_board[int(x)][int(y)] == 0 or self.previous_points.contains((int(x), int(y)))
        if not bool:
            print(self.previous_points)
            print(x, y)
            print(3)
        return bool

    def in_bounds(self, x_dest, y_dest):
        return 0 < x_dest < SCREEN_WIDTH and 0 < y_dest < SCREEN_HEIGHT

# ------------------------------------------------------------------------------
# -------------------------------              ---------------------------------
# -----------------------------   AchtungGame    ------------------------------
# -------------------------------              ---------------------------------
# ------------------------------------------------------------------------------

class AchtungGame:

    def __init__(self, number_of_players):
        self.game_board = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.players = []
        for i in range(number_of_players):
            new_player = Player(self.players, i+1)
            self.players.append(new_player)
        # self.players[0].x = 100
        # self.players[0].y = 50
        # self.players[1].x = 800
        # self.players[1].y = 50
        # self.players[0].theta = 0.25 * np.pi
        # self.players[1].theta = 0.75 * np.pi
        self.game_over = False
        #self.players = np.random.rand(number_of_players,3)


    # def set_players(number_of_players):
    #     for i in range(number_of_players):
    #         self.players.append(Player(self.players))


    def step(self, actions):
        # action = [player, theta_change]
        # theta_change = {-1:turn left, 0:dont change, 1:turn right}

        game_over = False
        #filters only actions for players that are still alive
        actions = [action for action in actions if action[0] in self.players]
        for player, theta_change in actions:
            is_player_still_alive = player.update(self.game_board, theta_change)
            #print(is_player_still_alive)
            if not is_player_still_alive:
                    self.players.remove(player)
        # for p in self.players:
            # print(p)
        if len(self.players) <= 1:
            game_over = True
        return self.game_board, game_over


    def __str__(self):
        pass

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
        self.screen = pygame.display.set_mode([SCREEN_WIDTH // 4, SCREEN_HEIGHT // 4])
        self.screen.fill((0, 0, 0))
        # ---- pygame end ----

    # -2: game-over, -1: go-left, 0: do-nothing, 1: go-right
    def get_input_from_user(self):

        for event in pygame.event.get():
            if event.type == KEYDOWN:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return INPUT["game-over"]


        key_array = KEY_ARRAY
        keys_pressed = pygame.key.get_pressed()

        actions = {}
        for i in range(len(self.game.players)):
            actions[i+1] = INPUT["do-nothing"]

        for i,keys in enumerate(key_array):
            id = i + 1
            if keys_pressed[keys[0]]:
                actions[id] = INPUT["go-left"]
            if keys_pressed[keys[1]]:
                if actions[id] == INPUT["do-nothing"]:
                    actions[id] = INPUT["go-right"]
                else:
                    actions[id] = INPUT["do-nothing"]

        for keys in key_array:
            if keys[0] in actions and keys[1] in actions:
                actions.remove(keys[0])
                actions.remove(keys[1])
        return actions

    def run_game(self):

        while True:
            input = self.get_input_from_user()
            if input == INPUT["game-over"]:
                break

            self.next_steps = [[player, input[player.id]] for player in self.game.players] # assigns each player the operation it has to do now
            game_board, game_over = self.game.step(self.next_steps)
            if game_over:
                print("Game is over")
                break

            self.render_game()
            pygame.display.flip()
            pygame.time.wait(REFRESH_SPEED)

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
                        (int(p.previous_points.top()[0])//4 , int(p.previous_points.top()[1]) // 4),
                        CIRCLE_SIZE
                    )
            # pygame.draw.lines(
            #             self.screen,
            #             COLORS[p.id],
            #             False,d
            #             [(int(p.previous_points.top()[0]) // 4, int(p.previous_points.top()[1]) // 4), (int(p.x)//4 , int(p.y) // 4)],
            #             10
            #         )

runner = AchtungGameRunner(2)
runner.run_game()


# TODO: show the head of the snakes (pygame.sprite)
# TODO: make it an envitonment
