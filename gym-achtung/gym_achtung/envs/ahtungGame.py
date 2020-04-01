
# --------- Imports ---------
from consts.Consts import *
import numpy as np
import time
# ------ End Of Imports -----


# ------------------------------------------------------------------------------
# ----------------------------------        ------------------------------------
# --------------------------------   Player   ----------------------------------
# ----------------------------------        ------------------------------------
# ------------------------------------------------------------------------------

class Player:

    def __init__(self, other_players):
        seld.id = self._generate_id(other_players)
        # 0 for x, 1 for y
        self.x, self.y = self._generate_rnd_coor(other_players)
        self.previus_x = self.x
        self.previus_y = self.y
        self.theta =  np.random.random() * 2*np.pi

    def get(self, i):
        if i == 0:
            return self.x
        if i == 1:
            return self.y
        return None

    def _generate_id(self, other_players):
        all_ids_taken = [p.id for p in other_players]
        free_ids = [i for i in range(MAX_PLAYERS) if i not in all_ids_taken]
        return np.random.choice(free_ids)

    def _generate_rnd_coor(self, other_players):

        def check_coordinates(point, other_points)
            for other_point in other_points:
                curr_dist = np.linalg.norm(other_point - point)
                if curr_dist < MIN_DIST_BETWEEN_SNAKES_START:
                    return False
            return True

        all_coor_taken = [(p.x, p.y) for p in other_players]
        new_point = [np.random.randint(1 , SCREEN_WIDTH), np.random.randint(1 , SCREEN_HEIGHT)]

        while check_coordinates(new_point, all_coor_taken) is not True:
            new_point = ( np.random.randint(1 , SCREEN_WIDTH), np.random.randint(1 , SCREEN_HEIGHT))

        return new_point

    def go_on_path(self, xDestination, yDestination, delta_x, delta_y, check=True):
        if not check:
            self.previus_x = self.x
            self.previus_y = self.y

        temp_x, temp_y = self.x, self.y
        x_got_to_place, y_got_to_place = False, False
        while not x_got_to_place or not y_got_to_place:

            # actual step/check-step
            temp_x += delta_x
            temp_y += delta_y
            if not check:
                self.x = temp_x
                self.y = temp_y

            # runs checker or real player
            if check:
                if not check_on_board(game_board, temp_x, temp_y):
                    return False
            else:
                draw_on_board(game_board, temp_x, temp_y)

            if int(temp_x) == xDestination:
                x_got_to_place = True
            if int(temp_y) == yDestination:
                y_got_to_place = True

        return True

    def update(self, game_board, theta_change):
        self.theta += theta_change * DELTA_THETA
        delta_x =  np.cos(self.theta)
        delta_y = np.sin(self.theta)
        xDestination = int(player.x + SPEED*delta_x)
        yDestination = int(player.y + SPEED*delta_y)

        # checks if destination is in the board
        if not self.in_bounds(xDestination, yDestination):
            return False

        # checks if path is ok
        if self.go_on_path(xDestination, yDestination, delta_x, delta_y, check=True):

            # This is a good move! goes on path
            self.go_on_path(xDestination, yDestination, delta_x, delta_y, check=False)
            return True

        # path is not ok - player is dead
        return False

    def draw_on_board(self, game_board, x, y):
        for i in range(-1, 2):
            game_board[int(x) + i, int(y)] = player.id
            game_board[int(x), int(y) + i] = player.id

    def check_on_board(self, game_board, x, y):
        if abs(int(x) - int(self.x)) <= 1 or abs(int(y) - int(self.y)) <= 1:
            return True
        return game_board[int(x), int(y)] == 0

    def in_bounds(self, x_dest, y_dest):
        return 0 < x_dest < SCREEN_WIDTH and 0 < y_dest < SCREEN_HEIGHT

# ------------------------------------------------------------------------------
# -------------------------------              ---------------------------------
# -----------------------------   AchtungGame    ------------------------------
# -------------------------------              ---------------------------------
# ------------------------------------------------------------------------------

class AchtungGame:

    def __init__(self, number_of_players):
        self.game_board = np.zeros((SCREEN_WIDTH,SCREEN_HEIGHT))
        self.players = []
        self.set_players(number_of_players)
        self.game_over = False
        #self.players = np.random.rand(number_of_players,3)


    def set_players(number_of_players):
        for i in range(number_of_players):
            self.players.append(Player(self.players))


    def step(self, actions):
        # action = [player, theta_change]
        # theta_change = {-1:turn left, 0:dont change, 1:turn right}

        game_over = False
        #filters only actions for players that are still alive
        actions = [action for action in actions if action[0] in self.players]
        for player, theta_change in actions:
            is_player_still_alive = player.update(game_board, theta_change)
            if not is_player_still_alive:
                    self.players  = self.players.remove(player)
        if len(self.players) <= 1:
            game_over = True
        return game_board, game_over


    def __str__(self):
        pass

# ------------------------------------------------------------------------------
# --------------------------------            ----------------------------------
# ------------------------------   GameRunner    -------------------------------
# --------------------------------            ----------------------------------
# ------------------------------------------------------------------------------

import pygame

class GameRunner:

    def __init__(self, num_of_players):
        self.game = AchtungGame(num_of_players)
        self.next_steps = []

        # ------ pygame ------
        pygame.init()
        self.screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
        self.screen.fill((0, 0, 0))
        # ---- pygame end ----

    # -2: game-over, -1: go-left, 0: do-nothing, 1: go-right
    def get_input_from_user(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1
        return 0

    def update_next_steps(self, input):
        if not INPUT["go-left"] <= input <= INPUT["go-right"]:
            return
        self.next_steps = [[player,0] for player in self.game.players]
        self.next_steps[0][1] = input

    def run_game(self):
        while True:
            input = self.get_input_from_user()
            if input == INPUT["game-over"]:
                break

            self.update_next_steps(input)
            game_board, game_over = self.game.step(self.next_steps)
            if game_over:
                break

            self.render_game()

            time.sleep(REFRESH_SPEED)

    def render_game(self):
        for p in self.game.players:
            pygame.draw.line(self.screen,
                        COLORS[p.id], (p.previus_x, p.previus_y), (p.x, p.y))


runner = GameRunner(2)
runner.run_game()

# TODO: Finish the render and run the game to see if it works
# TODO: fix all errors
# TODO:
