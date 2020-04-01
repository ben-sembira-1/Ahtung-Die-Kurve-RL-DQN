import numpy as np

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400
MAX_PLAYERS = 2
SPEED = 30
REFRESH_SPEED = 100
DELTA_THETA = 0.2 * np.pi
MIN_DIST_BETWEEN_SNAKES_START = 40

INPUT = {
    "go-right" : 1,
    "do-nothing" : 0,
    "go-left" : -1,
    "game-over" : -2,
}
