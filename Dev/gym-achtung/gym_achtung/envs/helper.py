from gym_achtung.envs.consts import *


class TmpQueue:
    def __init__(self, max_size=DEFAULT_MAX_QUEUE_SIZE):
        self.queue_list = []
        self.max_size = max_size
        self.size = len(self.queue_list)

    def push(self, e):
        if e is not None:
            if self.size >= self.max_size:
                self.queue_list = self.queue_list[:self.max_size]
            self.queue_list = [e] + self.queue_list
        self.size = len(self.queue_list)

    def top(self):
        return self.queue_list[0]

    def __str__(self):
        return str(self.queue_list)

    def contains(self, e):
        return e in self.queue_list
# import numpy as np
#
# new_board = np.ones((20,20))
#
# observation_space = np.zeros((1, 20, 20, 2))
#
# observation_space[:,:,:,1] =observation_space[:,:,:,0]
# observation_space[:,:,:,0] = np.expand_dims(new_board,axis=0)
#
