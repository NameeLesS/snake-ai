import torch
import numpy as np
from torch import nn
from torch.multiprocessing import Process, Value, Pipe

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
EPOCHS = 200
TARGET_UPDATE_FREQUENCY = 100

# Memory constants
MEMORY_SIZE = 200
ALPHA = 0.9
BETA = 0.4

possible_actions = [0, 1, 2, 3]


class TrainingProcess:
    def __init__(self, predict_network, target_network, device):
        self.predict_network = predict_network
        self.target_network = target_network
        self.device = device
        self.optimizer = torch.optim.AdamW(self.predict_network.parameters(), lr=LR, amsgrad=True)
        self.metrics = TrainMatrics()

    def training_step(self, gamma, data_update, samples):
        self.predict_network.train()
        self.target_network.train()

        if samples.poll(timeout=999):
            batch, weights, tree_idxs = samples.recv()
        states, next_states, actions, rewards, terminated = batch
        states = states.to(torch.float32).to(self.device)
        next_states = next_states.to(torch.float32).to(self.device)
        actions = actions.to(torch.int64).to(self.device)
        rewards = rewards.to(self.device)
        terminated = terminated.to(self.device)

        q_values = self.predict_network(states).gather(1, actions)

        with torch.no_grad():
            next_state_q_values = self.target_network(next_states).max(axis=1)[0]
        target_q_values = rewards + (1 - terminated) * gamma * next_state_q_values

        huber = nn.SmoothL1Loss(reduce=lambda x: torch.mean(x * weights))
        loss = huber(q_values, target_q_values.unsqueeze(1))
        td_error = torch.abs(q_values.squeeze(1) - target_q_values)
        data_update.send({'idxs': tree_idxs, 'priorities': td_error.detach()})

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.predict_network.parameters(), 100)
        self.optimizer.step()

        self.metrics.push_loss(loss.item())

    def training_loop(self, epochs, batch_size, data_update, samples, memory_size, sample_req, terminated):
        while int(memory_size.value) < batch_size:
            pass

        for epoch in range(epochs):
            print(f'======== {epoch + 1}/{epochs} epoch ========')
            sample_req.value = True
            self.training_step(GAMMA, data_update, samples)
            self.metrics.calculate()

            print(f'Loss: {self.metrics.loss}')

            if epoch % TARGET_UPDATE_FREQUENCY == 0:
                self.target_network.load_state_dict(self.predict_network.state_dict())

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

        self.metrics = TrainMatrics()

        self.data = data
        self.data_updates = data_updates
        self.samples = samples
        self.terminated = terminated
        self.memory_size = memory_size
        self.sample_req = sample_req

    def update_priorities(self):
        if self.data_updates.poll():
            update = self.data_updates.recv()
            self.memory.update_priorities(data_idxs=update['idxs'], priorities=update['priorities'])

    def update_data(self):
        if self.data.poll(timeout=999):
            data = self.data.recv()
            self.memory.add(data)
            self.metrics.push_reward(data[3], data[4])
            self.metrics.calculate()

    def send_sample(self):
        if self.sample_req.value:
            self.samples.send(self.memory.sample(BATCH_SIZE))
            self.sample_req.value = False

    def save_metrics(self):
        self.metrics.save('', 'metrics.pkl')

    def run(self):
        while True:
            self.memory_size.value = len(self.memory)

            self.update_data()
            self.update_priorities()
            self.send_sample()

            if self.terminated.value:
                break

        self.save_metrics()


class DataCollectionProcess(Process):
    def __init__(self, network, data, device, terminated):
        Process.__init__(self)
        self.predict_network = network
        self.data = data
        self.terminated = terminated
        self.device = device
        self.game = GameEnviroment(SCREEN_SIZE, FPS, False)

    def epsilon_greedy_policy(self, epsilon, state):
        if np.random.rand() < epsilon:
            return np.random.choice(possible_actions)
        else:
            self.predict_network.eval()

            with torch.no_grad():
                action = self.predict_network(state)

            return torch.argmax(action).to(torch.int)

    def do_one_step(self, epsilon):
        state = torch.tensor(self.game.get_state()).unsqueeze(1).reshape((1, *STATE_DIM)).to(torch.float32).to(
            self.device)
        action = self.epsilon_greedy_policy(epsilon, state)

        reward, next_state, terminated = self.game.step(int(action))

        reward = torch.tensor(reward)
        terminated = torch.tensor(terminated)
        next_state = torch.tensor(next_state).unsqueeze(1).reshape(STATE_DIM)

        data = (state, next_state, action, reward, terminated)
        return data

    def run(self):
        self.game.execute()
        while True:
            if self.terminated.value:
                break

            data = self.do_one_step(0.1)
            self.data.send(data)


def main():
    torch.multiprocessing.set_start_method('spawn', force=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predict_network = DQN().to(device)
    predict_network.share_memory()
    target_network = DQN().to(device)
    target_network.load_state_dict(predict_network.state_dict())

    training_process = TrainingProcess(
        predict_network=predict_network,
        target_network=target_network,
        device=device)

    data_recv, data_sender = Pipe()
    data_updates_recv, data_updated_sender = Pipe()
    samples_recv, samples_sender = Pipe()
    terminated = Value('b', False)
    memory_size = Value('i', 0)
    sample_req = Value('b', False)

    data_collection_process = DataCollectionProcess(
        predict_network,
        data_sender,
        device,
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

    training_process.training_loop(EPOCHS,
                                   BATCH_SIZE,
                                   data_updated_sender,
                                   samples_recv,
                                   memory_size,
                                   sample_req,
                                   terminated)


if __name__ == '__main__':
    main()
