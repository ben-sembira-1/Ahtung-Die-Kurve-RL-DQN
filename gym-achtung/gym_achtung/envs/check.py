from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_achtung.envs.consts import *


# -----------------------------------------
action_space = spaces.Discrete(3)
scaling = 4

high = np.array([SCREEN_WIDTH//scaling, SCREEN_HEIGHT//scaling]) # related to the 2-nd method below
low = np.zeros(2)
one_player_location_space = spaces.Box(low, high, dtype=np.int16)
# high = np.array([SCREEN_WIDTH//self.scaling, SCREEN_HEIGHT//self.scaling]*FooEnv.number_of_players)
# low = np.zeros(2 * FooEnv.number_of_players)
# all_players_location_space = spaces.box(low, high, dtype=np.int16)

# 0,1 1d array in size SCREEN_WIDTH/t X SCREEN_HEIGHT/t
game_board_space = spaces.MultiBinary(SCREEN_WIDTH//scaling * SCREEN_HEIGHT//scaling)
# 0,1 1d array in size SCREEN_WIDTH/t X SCREEN_HEIGHT/t box of all players with dtype=np.int16
observation_space = spaces.Tuple((one_player_location_space, game_board_space))

print(observation_space)
