from consts import *

class TmpHeap:
    def __init__(self, max_size=DEFAULT_MAX_HEAP_SIZE):
        self.heap_list = []
        self.max_size = max_size
        self.size = len(self.heap_list)

    def push(self, e):
        if e is not None:
            if self.size >= self.max_size:
                self.heap_list = self.heap_list[:self.max_size]
            self.heap_list = [e] + self.heap_list
        self.size = len(self.heap_list)

    def top(self):
        return self.heap_list[0]

    def __str__(self):
        return str(self.heap_list)

    def contains(self, e):
        return e in self.heap_list
