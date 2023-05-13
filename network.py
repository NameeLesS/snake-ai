import torch
import numpy as np
import random
import torch.nn.functional as F
from torch import nn
from collections import deque, namedtuple

from game import GameEnviroment
from graphs import Graphs
from config import *

LR = 1e-4
EPSILON = 0.7
GAMMA = 0.99
BATCH_SIZE = 8
EPOCHS = 10000
TARGET_UPDATE_FREQUENCY = 100
MEMORY_SIZE = 200


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 7), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(5),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),

            nn.Flatten(),

            nn.Linear(in_features=3136, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=4),
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
memory = Memory(MEMORY_SIZE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

predict_network = DQN().to(device)
target_network = DQN().to(device)
target_network.load_state_dict(predict_network.state_dict())

optimizer = torch.optim.AdamW(predict_network.parameters(), lr=LR, amsgrad=True)
possible_actions = [0, 1, 2, 3]

game = GameEnviroment(SCREEN_SIZE, FPS, False)
game.execute()
graphs = Graphs()


def epsilon_greedy_policy(epsilon, state):
    c = np.random.uniform(0, 1)
    if c < 1 - epsilon:
        return np.random.choice(possible_actions)
    else:
        target_network.eval()

        with torch.no_grad():
            action = predict_network(state)

        return int(torch.argmax(action))


def do_one_step():
    state = torch.tensor(game.get_state()[np.newaxis].reshape((1, 1, 400, 400))).to(device)
    action = epsilon_greedy_policy(EPSILON, state.to(torch.float32))

    reward, next_state, terminated = game.step(action)
    next_state = torch.tensor(next_state[np.newaxis].reshape((1, 1, 400, 400)))

    graphs.push_reward(reward, terminated)
    memory.push(Transition(state, next_state, action, reward, terminated))
    return state, next_state, action, reward, terminated


def training_step(batch_size, gamma):
    if len(memory) < batch_size:
        return

    states, next_states, actions, rewards, terminated = Transition(*zip(*memory.sample(batch_size)))
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    terminated = torch.tensor(terminated).to(device)
    states = torch.stack(states).squeeze(1).to(device).to(torch.float32)
    next_states = torch.stack(next_states).squeeze(1).to(device).to(torch.float32)

    mask = F.one_hot(actions, len(possible_actions))
    q_values = predict_network(states)
    q_values = torch.sum(q_values * mask, dim=1)

    with torch.no_grad():
        next_state_q_values = target_network(next_states).max(axis=1).values
        target_q_values = rewards + (1 - terminated) * gamma * next_state_q_values

    huber_loss = nn.SmoothL1Loss()
    loss = huber_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(predict_network.parameters(), 100)
    optimizer.step()

    graphs.push_loss(loss.item())
    print(f'Loss: {loss.item()}')


def training_loop(epochs, batch_size):
    for epoch in range(epochs):
        print(f'======== {epoch + 1}/{epochs} epoch')
        do_one_step()
        training_step(batch_size, GAMMA)

        if epoch % TARGET_UPDATE_FREQUENCY == 0:
            target_network.load_state_dict(predict_network.state_dict())

        print(graphs.get_series_reward())
        # graphs.plot_rewards()
        # graphs.plot_loss()


training_loop(EPOCHS, BATCH_SIZE)
