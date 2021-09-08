from gym_achtung.envs.consts import *
from scipy import ndimage


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