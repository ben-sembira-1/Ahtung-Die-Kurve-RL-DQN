import numpy as np
from pygame.locals import K_LEFT, K_RIGHT, K_ESCAPE, K_a, K_d, K_t, K_y, KEYDOWN, QUIT


SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1600
MAX_PLAYERS = 3
SPEED = 1
REFRESH_SPEED = 5
DELTA_THETA = 0.005 * np.pi
MIN_DIST_BETWEEN_SNAKES_START = 40
MAKE_GAP_PERIOD = 5
MAKE_GAP_DURATION = 1
GAP_EPSILON = 0.001
INPUT = {
    "go-right" : 1,
    "do-nothing" : 0,
    "go-left" : -1,
    "game-over" : -2,
}

CIRCLE_SIZE = 4
COLORS = {
    1 : (200, 250, 200),
    2 : (0, 200, 118),
    3 : (50, 100, 255)
}

KEY_ARRAY = [(K_LEFT, K_RIGHT), (K_a, K_d), (K_t, K_y)]

# Heap
DEFAULT_MAX_HEAP_SIZE = 70
