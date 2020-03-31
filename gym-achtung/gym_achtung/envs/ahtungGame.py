from consts.Consts import *
import numpy as np

class Player:

    def __init__(self, other_players):
        seld.id = self._generate_id(other_players)
        # 0 for x, 1 for y
        self.x = self._generate_rnd_coor(other_players, 0)
        self.y = self._generate_rnd_coor(other_players, 1)
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

    def _generate_rnd_coor(self, other_players, coor):

        def min_diatance(list, e):
            if not list: return Integer.MAX_VALUE
            min = abs(e - list[0])
            for list_e in list:
                curr_dist = abs(e - list_e)
                if curr_dist < min: min  = curr_dist
            return min

        screen_dims = [SCREEN_WIDTH, SCREEN_HEIGHT]
        all_coor_taken = [p.get(coor) for p in other_players]
        free_coor = [d for d in range(screen_dims[coor]) if min_diatance(all_coor_taken, d)]

    def move(self, theta):
    # TODO: move to here code for moving snake
        for player in self.players:
            gameBoard[player.x, player.y] = player.id
            delta_x =  np.cos(theta)
            delta_y = np.sin(theta)
            x, y = player.x, player.y
            xDestination, yDestination = player.x + SPEED*np.cos(theta), player.y + SPEED*np.sin(theta)
            for i in range(SPEED):
                x = ceil(x + delta_x)
                y = ceil(y + delta_y)
                gameBoard[x,y] = player.id
                if x > xDestination or y > yDestination:
                    break
        pass


class AhtungGame:

    def __init__(self, numberOfPlayers):
        self.gameBoard = np.zeros((SCREEN_WIDTH,SCREEN_HEIGHT))
        self.players = self.set_random_places(numberOfPlayers)
        #self.players = np.random.rand(numberOfPlayers,3)


    def set_random_places(numberOfPlayers):
        list_of_players = []
        for i in range numberOfPlayers:
            rnd_x = np.random.rndint(0,SCREEN_WIDTH)
            rnd_y = np.random.rndint(0,SCREEN_HEIGHT)
            rnd_theta = np.random.
            while self.
            list_of_players.

    def step():




    def __str__(self):
        pass

    def play_game(self):
        pass
