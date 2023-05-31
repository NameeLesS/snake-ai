import torch
import numpy as np
import traceback
from torch import nn
from multiprocessing import Process, Manager, Pipe

from game import GameEnviroment
from memory import PrioritizedReplayBuffer
from metrics import TrainMatrics
from config import *

# General constants
STATE_DIM = (1, SCREEN_SIZE[0], SCREEN_SIZE[1])

# Training constants
LR = 1e-4
GAMMA = 0.99
BATCH_SIZE = 8
EPOCHS = 50
TARGET_UPDATE_FREQUENCY = 100
STEPS = 50

# Memory constants
MEMORY_SIZE = 200
ALPHA = 0.9
BETA = 0.4


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(7, 7), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(5),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),

            nn.Flatten(),

            nn.Linear(in_features=1024, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=4),
        )

    def forward(self, x):
        x = x / 255
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
metrics = TrainMatrics()


def training_step(batch_size, gamma):
    if len(memory) < batch_size:
        return

    predict_network.train()
    target_network.train()

    batch, weights, tree_idxs = memory.sample(batch_size)
    states, next_states, actions, rewards, terminated = batch
    states = states.to(torch.float32).to(device)
    next_states = next_states.to(torch.float32).to(device)
    actions = actions.to(torch.int64).to(device)
    rewards = rewards.to(device)
    terminated = terminated.to(device)

    q_values = predict_network(states).gather(1, actions)

    with torch.no_grad():
        next_state_q_values = target_network(next_states).max(axis=1)[0]
    target_q_values = rewards + (1 - terminated) * gamma * next_state_q_values

    huber = nn.SmoothL1Loss(reduce=lambda x: torch.mean(x * weights))
    loss = huber(q_values, target_q_values.unsqueeze(1))
    td_error = torch.abs(q_values.squeeze(1) - target_q_values)
    memory.update_priorities(data_idxs=tree_idxs, priorities=td_error)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(predict_network.parameters(), 100)
    optimizer.step()

    metrics.push_loss(loss.item())


def training_loop(epochs, batch_size, network, data_receiver, terminated):
    for epoch in range(epochs):
        print(f'======== {epoch + 1}/{epochs} epoch ========')
        training_step(batch_size, GAMMA)
        metrics.calculate()

        print(f'Loss: {metrics.loss}')

        if epoch % TARGET_UPDATE_FREQUENCY == 0:
            target_network.load_state_dict(predict_network.state_dict())

        for step in range(STEPS):
            memory.add(data_receiver.recv())
        network.clear()
        network.update(predict_network.state_dict())
    terminated.value = True


class DataCollectionProcess(Process):
    def __init__(self, network, data_sender, terminated):
        Process.__init__(self)
        self.network = network
        self.data_sender = data_sender
        self.terminated = terminated
        self.game = GameEnviroment(SCREEN_SIZE, FPS, False)
        self.predict_network = DQN()

    def epsilon_greedy_policy(self, epsilon, state):
        if np.random.rand() < epsilon:
            return np.random.choice(possible_actions)
        else:
            self.predict_network.eval()

            with torch.no_grad():
                action = self.predict_network(state)

            return torch.argmax(action).to(torch.int)

    def do_one_step(self, epsilon):
        state = torch.tensor(self.game.get_state()).unsqueeze(1).reshape((1, *STATE_DIM)).to(torch.float32).to(device)
        action = self.epsilon_greedy_policy(epsilon, state)

        reward, next_state, terminated = self.game.step(int(action))

        reward = torch.tensor(reward)
        terminated = torch.tensor(terminated)
        next_state = torch.tensor(next_state).unsqueeze(1).reshape(STATE_DIM)

        metrics.push_reward(reward, terminated)
        data = (state, next_state, action, reward, terminated)
        return data, terminated

    def run(self):
        try:
            self.game.execute()
            while True:
                if self.terminated.value:
                    break
                if self.network:
                    self.predict_network.load_state_dict(self.network)
                data, terminated = self.do_one_step(0.1)
                self.data_sender.send(data)
            print('end')
        except:
            traceback.print_exc()


def main():
    with Manager() as manager:
        network = manager.dict(predict_network.state_dict())
        terminated = manager.Value('b', False)
        data_recv, data_sender = Pipe()

        data_collection_process = DataCollectionProcess(network, data_sender, terminated)
        data_collection_process.start()
        training_loop(EPOCHS, BATCH_SIZE, network, data_recv, terminated)


if __name__ == '__main__':
    main()
