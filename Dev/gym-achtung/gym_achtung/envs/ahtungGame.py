# --------- Imports ---------
from gym_achtung.envs.consts import *
import numpy as np
from gym_achtung.envs.player import Player

# ------ End Of Imports -----

# ------ Constants ------





# ------------------------------------------------------------------------------
# ----------------------------------        ------------------------------------
# --------------------------------   Player   ----------------------------------
# ----------------------------------        ------------------------------------
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# -------------------------------              ---------------------------------
# -----------------------------   AchtungGame    ------------------------------
# -------------------------------              ---------------------------------
# ------------------------------------------------------------------------------

class AchtungGame:
    num_of_turns = 0
    SCREEN_WIDTH = 400
    SCREEN_HEIGHT = 400

    def __init__(self, number_of_players=NUMBER_OF_PLAYERS):
        self.game_board = np.zeros((AchtungGame.SCREEN_WIDTH, AchtungGame.SCREEN_HEIGHT))
        self.players = []
        for i in range(number_of_players):
            self.players.append(Player(i + 1, self._generate_rnd_coor(self.players)))
        self.game_over = False

    def _generate_rnd_coor(self, other_players):

        if other_players == []:
            new_point = [np.random.randint( int(AchtungGame.SCREEN_WIDTH//4), int(3*AchtungGame.SCREEN_WIDTH//4)), np.random.randint(int(AchtungGame.SCREEN_HEIGHT//4), int(3*AchtungGame.SCREEN_HEIGHT//4))]
            return new_point

        def check_coordinates(point, other_points):
            for other_point in other_points:
                curr_dist = np.linalg.norm([other_point[0] - point[0],other_point[1] - point[1]])
                if curr_dist < MIN_DIST_BETWEEN_SNAKES_START:
                    return False
            return True

        all_coor_taken = [[p.x, p.y] for p in other_players]
        new_point = [np.random.randint( int(AchtungGame.SCREEN_WIDTH//4), int(3*AchtungGame.SCREEN_WIDTH//4)), np.random.randint(int(AchtungGame.SCREEN_HEIGHT//4), int(3*AchtungGame.SCREEN_HEIGHT//4))]

        while check_coordinates(new_point, all_coor_taken) is not True:
            new_point = [np.random.randint( int(AchtungGame.SCREEN_WIDTH//4), int(3*AchtungGame.SCREEN_WIDTH//4)), np.random.randint(int(AchtungGame.SCREEN_HEIGHT//4), int(3*AchtungGame.SCREEN_HEIGHT//4))]

        return new_point

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


# TODO: fix gaps to be based on turns and not time
# TODO: first priority - improve existing player
# TODO: one method:
