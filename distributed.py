import multiprocessing.connection

import torch
import numpy as np
import traceback
from torch import nn
from multiprocessing import Process, Value, Pipe

from game import GameEnviroment
from memory import PrioritizedReplayBuffer
from metrics import TrainMatrics
from network import DQN
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
predict_network = DQN().to(device)
target_network = DQN().to(device)
target_network.load_state_dict(predict_network.state_dict())

optimizer = torch.optim.AdamW(predict_network.parameters(), lr=LR, amsgrad=True)
possible_actions = [0, 1, 2, 3]

metrics = TrainMatrics()


def training_step(batch_size, gamma, data_update, samples):
    predict_network.train()
    target_network.train()
    ready_connections = multiprocessing.connection.wait([samples])

    if samples in ready_connections:
        batch, weights, tree_idxs = samples.recv()
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
    data_update.send({'idxs': tree_idxs, 'priorities': td_error})

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(predict_network.parameters(), 100)
    optimizer.step()

    metrics.push_loss(loss.item())


def training_loop(epochs, batch_size, network, data_update, samples, memory_size, sample_req, terminated):
    while int(memory_size.value) < batch_size:
        pass

    for epoch in range(epochs):
        print(f'======== {epoch + 1}/{epochs} epoch ========')
        sample_req.value = True
        training_step(batch_size, GAMMA, data_update, samples)
        metrics.calculate()

        print(f'Loss: {metrics.loss}')

        if epoch % TARGET_UPDATE_FREQUENCY == 0:
            target_network.load_state_dict(predict_network.state_dict())

        network.send(predict_network.state_dict())

    terminated.value = True


class MemoryManagmentProcess(Process):
    def __init__(self, data, data_updates, samples, memory_size, sample_req, terminated):
        Process.__init__(self)
        self.memory = PrioritizedReplayBuffer(
            buffer_size=MEMORY_SIZE,
            state_dim=STATE_DIM,
            action_dim=(1),
            alpha=ALPHA, beta=BETA
        )

        self.data = data
        self.data_updates = data_updates
        self.samples = samples
        self.terminated = terminated
        self.memory_size = memory_size
        self.sample_req = sample_req

    def update_priorities(self):
        update = self.data_updates.recv()
        self.memory.update_priorities(data_idxs=update['idxs'], priorities=update['priorities'])

    def run(self):
        while True:
            self.memory_size.value = len(self.memory)
            if self.data.poll():
                self.memory.add(self.data.recv())
            if self.data_updates.poll():
                self.update_priorities()
            if self.sample_req.value:
                self.samples.send(self.memory.sample(BATCH_SIZE))
                self.sample_req.value = False
            if self.terminated.value:
                break
            print(f'memory managment {self.memory_size}')


class DataCollectionProcess(Process):
    def __init__(self, network, data, terminated):
        Process.__init__(self)
        self.network = network
        self.data = data
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
                if self.network.poll():
                    nt = self.network.recv()
                    self.predict_network.load_state_dict(nt)

                data, terminated = self.do_one_step(0.1)
                self.data.send(data)
                print('collection')
        except:
            traceback.print_exc()


def main():
    network_recv, network_sender = Pipe()
    data_recv, data_sender = Pipe()
    data_updates_recv, data_updated_sender = Pipe()
    samples_recv, samples_sender = Pipe()
    network_sender.send(predict_network.state_dict())
    terminated = Value('b', False)
    memory_size = Value('i', 0)
    sample_req = Value('b', False)

    data_collection_process = DataCollectionProcess(
        network_recv,
        data_sender,
        terminated
    )

    memory_managment_process = MemoryManagmentProcess(
        data_recv,
        data_updates_recv,
        samples_sender,
        memory_size,
        sample_req,
        terminated
    )

    data_collection_process.start()
    memory_managment_process.start()

    training_loop(EPOCHS,
                  BATCH_SIZE,
                  network_sender,
                  data_updated_sender,
                  samples_recv,
                  memory_size,
                  sample_req,
                  terminated)


if __name__ == '__main__':
    main()
