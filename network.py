import torch
import numpy as np
from torch import nn

from game import GameEnviroment
from graphs import Graphs
from memory import PrioritizedReplayBuffer
from config import *

# General constants
STATE_DIM = (1, 400, 400)

# Training constants
LR = 1e-4
EPSILON = 0.7
GAMMA = 0.99
BATCH_SIZE = 4
EPOCHS = 10000
TARGET_UPDATE_FREQUENCY = 100

# Memory constants
MEMORY_SIZE = 200
ALPHA = 0.9
BETA = 0.4


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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

memory = PrioritizedReplayBuffer(buffer_size=MEMORY_SIZE, state_dim=STATE_DIM, action_dim=(1), alpha=ALPHA, beta=BETA)
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

        return torch.argmax(action).to(torch.int)


def do_one_step():
    state = torch.tensor(game.get_state()).unsqueeze(1).reshape((1, *STATE_DIM)).to(device)
    action = epsilon_greedy_policy(EPSILON, state.to(torch.float32))

    reward, next_state, terminated = game.step(int(action))

    reward = torch.tensor(reward).to(device)
    terminated = torch.tensor(terminated).to(device)
    next_state = torch.tensor(next_state).unsqueeze(1).reshape(STATE_DIM)

    graphs.push_reward(reward, terminated)
    memory.add((state, next_state, action, reward, terminated))
    return state, next_state, action, reward, terminated


def training_step(batch_size, gamma):
    if len(memory) < batch_size:
        return

    batch, weights, tree_idxs = memory.sample(batch_size)
    states, next_states, actions, rewards, terminated = batch
    states, next_states = states.to(torch.float32), next_states.to(torch.float32)
    actions = actions.to(torch.int64)

    q_values = predict_network(states).gather(1, actions)

    with torch.no_grad():
        next_state_q_values = target_network(next_states).max(axis=1)[0]
    target_q_values = rewards + (1 - terminated) * gamma * next_state_q_values

    criterion = nn.SmoothL1Loss()
    loss = criterion(q_values, target_q_values.unsqueeze(1))

    td_error = torch.abs(q_values.squeeze(1) - target_q_values)
    memory.update_priorities(data_idxs=tree_idxs, priorities=td_error)

    optimizer.zero_grad()
    loss.backward()
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


training_loop(EPOCHS, BATCH_SIZE)
