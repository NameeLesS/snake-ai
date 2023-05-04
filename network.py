import torch
import random
from torch import nn
from collections import deque, namedtuple


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7, 7), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(5),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(3),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(3),

            nn.Flatten(),

            nn.Linear(in_features=131072, out_features=2048),
            nn.Linear(in_features=2048, out_features=4),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Memory:
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def push(self, element):
        self.memory.append(element)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


Transition = namedtuple('Transition', ('state', 'n_state', 'action', 'reward'))

# Step function should return:
# - current state
# - next state
# - reward
# enviroment should contain informations about:
# - Possible actions

