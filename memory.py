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
        self.data = np.zeros(capacity, dtype=object)
        self.capacity = capacity
        self.last_idx = 0
        self.size = 0

    def add(self, prority, data):
        idx = self.last_idx + self.capacity - 1
        self.data[self.last_idx] = data

        self.update(idx, prority)

        self.last_idx += 1
        if self.last_idx >= self.capacity:
            self.last_idx = 0

        if self.size < self.capacity:
            self.size += 1

    def update(self, idx, prority):
        change = prority - self.nodes[idx]
        self.nodes[idx] = prority

        self._propagate(idx, change)

    def get(self, sumcum):
        idx = self._retrieve(0, sumcum)
        data_idx = idx - self.capacity + 1
        return self.nodes[idx], self.data[data_idx]

    def _propagate(self, idx, change):
        parent = self.parent(idx)
        self.nodes[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, sumcum):
        left = self.left_child(idx)
        right = self.right_child(idx)

        if left >= 2 * self.capacity - 1:
            return idx

        if self.nodes[left] >= sumcum:
            return self._retrieve(left, sumcum)
        else:
            return self._retrieve(right, sumcum - self.nodes[left])

    @staticmethod
    def parent(idx):
        return (idx - 1) // 2

    @staticmethod
    def right_child(idx):
        return idx * 2 + 2

    @staticmethod
    def left_child(idx):
        return idx * 2 + 1

    @property
    def sumcum(self):
        return self.nodes[0]

    def __len__(self):
        return self.size
