from gym_achtung.envs.consts import *
from collections import deque


class Player:

    def __init__(self, player_id, coordinates=(0, 0)):
        self.id = player_id
        self.x, self.y = coordinates
        self.previous_points = deque(maxlen=DEFAULT_MAX_QUEUE_SIZE)
        self.previous_points.append((int(self.x), int(self.y)))
        self.theta = np.random.random() * 2 * np.pi
        self.turn_last_gap_started = 0
        self.curr_turn = 0
        self.is_gapping = False

    def __str__(self):
        return f"PLAYER: id-{self.id}, position-({self.x}, {self.y}), theta-{self.theta}"

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
        delta_x = np.cos(self.theta)
        delta_y = np.sin(self.theta)
        xDestination = int(self.x + SPEED * delta_x)
        yDestination = int(self.y + SPEED * delta_y)

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
