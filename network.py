import torch
import numpy as np
import random
from torch import nn
from collections import deque, namedtuple

from game import GameEnviroment
from graphs import Graphs
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
huber_loss = nn.HuberLoss()
optimizer = torch.optim.Adam(predict_network.parameters(), lr=0.01)
possible_actions = [0, 1, 2, 3]

game = GameEnviroment(SCREEN_SIZE, FPS)
game.execute()
graphs = Graphs()


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
    state = torch.tensor(game.get_state()[np.newaxis].reshape((1, 3, 800, 800)), dtype=torch.float32).to(device)
    action = epsilon_greedy_policy(0.9, state)

    reward, next_state, terminated = game.step(action)
    next_state = torch.tensor(next_state[np.newaxis].reshape((1, 3, 800, 800)), dtype=torch.float32)

    graphs.push_reward(reward)
    memory.push(Transition(state, next_state, action, reward, terminated))
    return state, next_state, action, reward, terminated


def training_step(batch_size, gamma):
    if len(memory) < batch_size:
        return

    states, next_states, actions, rewards, terminated = Transition(*zip(*memory.sample(batch_size)))
    rewards = torch.tensor(rewards).to(device)
    terminated = torch.tensor(terminated).to(device)
    states = torch.stack(states).squeeze(1).to(device)

    estimated_q_values = predict_network(states).max(axis=1).values

    target_q_values = rewards + (1 - terminated) * gamma * estimated_q_values

    loss = huber_loss(estimated_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    target_network.load_state_dict(predict_network.state_dict())

    graphs.push_loss(loss.item())
    print(f'Loss: {loss.item()}')


def training_loop(epochs, batch_size):
    for epoch in range(epochs):
        print(f'======== {epoch + 1}/{epochs} epoch')
        do_one_step()
        training_step(batch_size, 0.9)
        graphs.plot_rewards()
        graphs.plot_loss()


training_loop(30, 1)
