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


# TODO: fix gaps to be based on turns and not time
# TODO: first priority - improve existing player
# TODO: one method:
