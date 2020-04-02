import numpy as np

SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1600
MAX_PLAYERS = 2
SPEED = 4
REFRESH_SPEED = 50
DELTA_THETA = 0.05 * np.pi
MIN_DIST_BETWEEN_SNAKES_START = 40

INPUT = {
    "go-right" : 1,
    "do-nothing" : 0,
    "go-left" : -1,
    "game-over" : -2,
}

COLORS = {
    1 : (200, 250, 200),
    2 : (150, 250, 118)
}
