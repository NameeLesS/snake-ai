import torch
import numpy as np
import random
from torch import nn
from collections import deque, namedtuple

from game import GameEnviroment
from config import *


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
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=4),
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


Transition = namedtuple('Transition', ('state', 'n_state', 'action', 'reward', 'terminated'))
memory = Memory(2000)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

predict_network = DQN().to(device)
target_network = DQN().to(device)
possible_actions = [0, 1, 2, 3]

game = GameEnviroment(SCREEN_SIZE, FPS)
game.execute()


def epsilon_greedy_policy(epsilon, state):
    c = np.random.uniform(0, 1)
    if c < 1 - epsilon:
        return np.random.choice(possible_actions)
    else:
        target_network.eval()

        with torch.no_grad():
            action = target_network(state)

        return int(torch.argmax(action))


def do_one_step():
    state = torch.tensor(game.get_state()[np.newaxis].reshape((1, 3, 800, 800)), dtype=torch.float32)
    action = epsilon_greedy_policy(0.9, state)

    reward, next_state, terminated = game.step(action)
    next_state = torch.tensor(next_state[np.newaxis].reshape((1, 3, 800, 800)), dtype=torch.float32)

    memory.push(Transition(state, next_state, action, reward, terminated))
    return state, next_state, action, reward, terminated


def training_step(batch_size, gamma):
    states, next_states, actions, rewards, terminated = Transition(*zip(*memory.sample(batch_size)))
    rewards = torch.tensor(rewards)
    terminated = torch.tensor(terminated)
    states = torch.stack(states).squeeze(1)

    max_q_values = predict_network(states).gather(-1, torch.tensor([actions]))

    q_values = rewards + (1 - terminated) * gamma * max_q_values


for i in range(10):
    do_one_step()

training_step(1, 0.9)
