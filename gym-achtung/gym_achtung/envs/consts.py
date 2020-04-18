import numpy as np
from pygame.locals import K_LEFT, K_RIGHT, K_ESCAPE, K_a, K_d, K_t, K_y, KEYDOWN, QUIT

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
SHOW_BOARD = False
FINAL_SIZE = 20
RADIUS = 3



MAX_PLAYERS = 1
SCALING = 3
ZOOM = 2
SPEED = 1
REFRESH_SPEED = 30
DELTA_THETA = 0.005 * np.pi
MIN_DIST_BETWEEN_SNAKES_START = 40
MAKE_GAP_PERIOD = 5
MAKE_GAP_DURATION = 0.6
GAP_EPSILON = 0.001
NUMBER_OF_TURNS_TO_LIVE = 50000
WIN_REWARD = 70000


DQN_EPSIOLN = 0.7
DQN_GAMMA = 0.85
EPISODES = 5000


INPUT = {
    "go-right" : 2,
    "do-nothing" : 0,
    "go-left" : 1,
    "game-over" : -2,
}
FROM_INPUT_TO_THETA_CHANGE = {
    INPUT["go-right"] : 1,
    INPUT["do-nothing"] : 0,
    INPUT["go-left"] : -1,
}

CIRCLE_SIZE = 4
COLORS = {
    1: (200, 250, 200),
    2: (0, 200, 118),
    3: (50, 100, 255)
}

KEY_ARRAY = [(K_LEFT, K_RIGHT), (K_a, K_d), (K_t, K_y)]

# Queue
DEFAULT_MAX_QUEUE_SIZE = 70
