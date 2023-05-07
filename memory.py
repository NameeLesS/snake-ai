import random
import torch
import numpy as np

from torch import nn
from collections import deque, namedtuple


class PrioritizedReplayBuffer:
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def push(self, element):
        self.memory.append(element)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SumTree:
    def __init__(self, capacity):
        self.nodes = np.zeros(2 * capacity - 1)
        self.data = np.zeros(2 * capacity, dtype=object)
        self.capacity = capacity
        self.size = 0
        self.count = 0

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.size = min(self.capacity, self.size + 1)


    def update(self, idx, value):
        pass

    def get(self):
        pass

    def _retrieve(self):
        pass

    def __len__(self):
        pass


SumTree(10)
