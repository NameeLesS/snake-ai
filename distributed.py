import torch
import numpy as np
import os
from torch import nn
from torch.multiprocessing import (
    Process,
    Value,
    Pipe,
    Array,
    Event,
    RLock
)

from game import GameEnviroment
from memory import PrioritizedReplayBuffer
from metrics import TrainMatrics
from network import DQN
from config import *

import traceback

# General constants
STATE_DIM = (1, SCREEN_SIZE[0], SCREEN_SIZE[1])

# Training constants
LR = 1e-4
GAMMA = 0.99
BATCH_SIZE = 64
EPOCHS = 50000
TARGET_UPDATE_FREQUENCY = 2000
STEPS = 4
DECAY_RATE_CHANGE = 7e-6

# Memory constants
MEMORY_SIZE = 100000
ALPHA = 0.9
BETA = 0.4

possible_actions = [0, 1, 2, 3]


class TrainingProcess:
    def __init__(
            self,
            predict_network,
            target_network,
            device,
            metrics,
            network,
            data_updates,
            samples,
            memory_size,
            sample_request_event,
            loss,
            terminated,
            model_path=None,
            save_path=''
    ):
        self.predict_network = predict_network
        self.target_network = target_network
        self.device = device
        self.metrics = metrics
        self.network = network
        self.data_updates = data_updates
        self.samples = samples
        self.memory_size = memory_size
        self.sample_request_event = sample_request_event
        self.loss = loss
        self.terminated = terminated
        self.model_path = model_path
        self.save_path = save_path

        self.optimizer = torch.optim.AdamW(self.predict_network.parameters(), lr=LR, amsgrad=True)

    def training_step(self, gamma, sample):
        self.predict_network.train()
        self.target_network.eval()

        batch, weights, tree_idxs = sample
        states, next_states, actions, rewards, terminated = batch
        states = states.to(torch.float32).to(self.device)
        next_states = next_states.to(torch.float32).to(self.device)
        actions = actions.to(torch.int64).to(self.device)
        rewards = rewards.to(self.device)
        terminated = terminated.to(self.device)

        try:
            q_values = self.predict_network(states).gather(1, actions)
        except Exception as e:
            traceback.print_exc()

        with torch.no_grad():
            next_state_q_values = self.target_network(next_states).max(axis=1)[0]
        target_q_values = rewards + (1 - terminated) * gamma * next_state_q_values

        huber = nn.SmoothL1Loss(reduce=lambda x: torch.mean(x * weights))
        loss = huber(q_values, target_q_values.unsqueeze(1))
        td_error = torch.abs(q_values.squeeze(1) - target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.predict_network.parameters(), 100)
        self.optimizer.step()

        return tree_idxs, td_error, loss.item()

    def training_loop(self, epochs, batch_size, gamma):
        while int(self.memory_size.value) < batch_size:
            pass

        for epoch in range(epochs):
            print(f'======== {epoch + 1}/{epochs} epoch ========')
            self.sample_request_event.set()
            if self.samples.poll(timeout=999):
                sample = self.samples.recv()

                tree_idxs, td_error, loss = self.training_step(gamma, sample)

                print(
                    f'Loss: {loss} '
                    f'Average rewards: {self.metrics[0]} '
                    f'Average episode length: {self.metrics[1]}')
                print(
                    f'Highest reward: {self.metrics[2]} '
                    f'Longest episode: {self.metrics[3]}')

                if epoch % TARGET_UPDATE_FREQUENCY == 0:
                    self.target_network.load_state_dict(self.predict_network.state_dict())

                self.data_updates.send({'idxs': tree_idxs, 'priorities': td_error.detach()})
                self.loss.send(loss)
                self.network.send({
                    'epoch': epoch,
                    'network': self.predict_network.state_dict()
                })
        self.terminated.value = True
        self.save_model()

    def save_model(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.predict_network.state_dict(), os.path.join(self.save_path, 'model.pth'))

    def load_model(self):
        if self.model_path is not None:
            self.predict_network.load_state_dict(torch.load(os.path.join(self.model_path, 'model.pth')))
            self.target_network.load_state_dict(self.predict_network.state_dict())
        else:
            print('Model not found, creating new one')


class MemoryManagmentProcess(Process):
    def __init__(
            self,
            data,
            metrics,
            data_updates,
            samples,
            memory_size,
            sample_request_event,
            loss,
            terminated,
            save_path=''
    ):
        Process.__init__(self)
        self.memory = None
        self.metrics = None

        self.data = data
        self.data_updates = data_updates
        self.samples = samples
        self.terminated = terminated
        self.memory_size = memory_size
        self.sample_request_event = sample_request_event
        self.loss = loss
        self.metrics_summary = metrics
        self.save_path = save_path

    def init(self):
        self.memory = PrioritizedReplayBuffer(
            buffer_size=MEMORY_SIZE,
            state_dim=INPUT_SIZE,
            action_dim=(1),
            alpha=ALPHA,
            beta=BETA
        )

        self.metrics = TrainMatrics()

    def update_priorities(self):
        if self.data_updates.poll():
            update = self.data_updates.recv()
            self.memory.update_priorities(data_idxs=update['idxs'], priorities=update['priorities'])

    def update_data(self):
        if self.data.poll(timeout=999):
            data = self.data.recv()
            self.memory.add(data)
            self.metrics.push_reward(data[3], data[4])

    def send_sample(self):
        if self.sample_request_event.is_set():
            self.samples.send(self.memory.sample(BATCH_SIZE))
            self.sample_request_event.clear()

    def update_metrics(self):
        if self.loss.poll():
            self.metrics.push_loss(self.loss.recv())

    def save_metrics(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.metrics.save(self.save_path, 'metrics')

    def run(self):
        self.init()
        try:
            while True:
                self.memory_size.value = len(self.memory)

                self.update_data()
                self.update_priorities()
                self.send_sample()
                self.update_metrics()

                self.metrics.calculate()
                self.metrics_summary[0] = self.metrics.average_rewards
                self.metrics_summary[1] = self.metrics.average_episode_length
                self.metrics_summary[2] = self.metrics.highest_reward
                self.metrics_summary[3] = self.metrics.longest_episode

                if self.terminated.value:
                    break

        except (Exception, KeyboardInterrupt) as e:
            self.save_metrics()
        self.save_metrics()


class DataCollectionProcess(Process):
    def __init__(
            self,
            network,
            data,
            device,
            data_collection_lock,
            terminated
    ):
        Process.__init__(self)
        self.predict_network = None
        self.network = network
        self.data = data
        self.terminated = terminated
        self.device = device
        self.data_collection_lock = data_collection_lock
        self.game = None

    def init(self):
        self.predict_network = DQN().to(self.device)
        self.game = GameEnviroment(SCREEN_SIZE, FPS, False)
        self.game.execute()

    def epsilon_greedy_policy(self, epsilon, state):
        if np.random.rand() < epsilon:
            return torch.tensor(np.random.choice(possible_actions))
        else:
            self.predict_network.eval()

            with torch.no_grad():
                action = self.predict_network(state)

            return torch.argmax(action).to(torch.int)

    def do_one_step(self, epsilon):
        state = self.preprocess_state(self.game.get_state()).to(torch.float32).to(self.device)

        action = self.epsilon_greedy_policy(epsilon, state)

        reward, next_state, terminated = self.game.step(int(action))

        reward = torch.tensor(reward)
        terminated = torch.tensor(terminated)
        next_state = self.preprocess_state(next_state).to(self.device)

        data = (state, next_state, action, reward, terminated)
        return data

    def preprocess_state(self, state):
        state = torch.tensor(state).unsqueeze(0).unsqueeze(0)
        state = torch.nn.functional.interpolate(state, INPUT_SIZE[1:3])
        state = state.squeeze().reshape((1, *INPUT_SIZE))
        return state

    def collect_data(self, steps):
        self.data_collection_lock.acquire()
        if self.network.poll(timeout=999):
            network = self.network.recv()
            self.data_collection_lock.release()
            print(f"Running data collection for epoch: {network['epoch']}")
            self.predict_network.load_state_dict(network['network'])
            for step in range(steps):
                epsilon = 1 * (1 - DECAY_RATE_CHANGE) ** network['epoch']
                data = self.do_one_step(epsilon)
                self.data.send(data)

    def run(self):
        self.init()
        self.collect_data(200)
        while True:
            if self.terminated.value:
                break

            self.collect_data(STEPS)


def main():
    torch.multiprocessing.set_start_method('spawn', force=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predict_network = DQN().to(device)
    target_network = DQN().to(device)
    target_network.load_state_dict(predict_network.state_dict())

    metrics = Array('d', 4)
    network_recv, network_sender = Pipe()
    data_recv, data_sender = Pipe()
    data_updates_recv, data_updated_sender = Pipe()
    samples_recv, samples_sender = Pipe()
    loss_recv, loss_sender = Pipe()
    sample_request_event = Event()
    data_collection_lock = RLock()
    terminated = Value('b', False)
    memory_size = Value('i', 0)

    network_sender.send({
        'epoch': 0,
        'network': predict_network.state_dict()
    })

    training_process = TrainingProcess(
        predict_network=predict_network,
        target_network=target_network,
        device=device,
        network=network_sender,
        data_updates=data_updated_sender,
        samples=samples_recv,
        memory_size=memory_size,
        sample_request_event=sample_request_event,
        loss=loss_sender,
        metrics=metrics,
        save_path='backups/models/',
        terminated=terminated
    )

    data_collection_processes = [DataCollectionProcess(
        network=network_recv,
        data=data_sender,
        device=device,
        data_collection_lock=data_collection_lock,
        terminated=terminated
    ) for _ in range(1)]

    memory_managment_process = MemoryManagmentProcess(
        data=data_recv,
        data_updates=data_updates_recv,
        samples=samples_sender,
        memory_size=memory_size,
        sample_request_event=sample_request_event,
        loss=loss_recv,
        metrics=metrics,
        save_path='backups/metrics/',
        terminated=terminated
    )

    for data_collection_process in data_collection_processes:
        data_collection_process.start()
    memory_managment_process.start()

    try:
        training_process.training_loop(EPOCHS, BATCH_SIZE, GAMMA)
    except (Exception, KeyboardInterrupt) as e:
        training_process.save_model()

    for data_collection_process in data_collection_processes:
        data_collection_process.join()
    memory_managment_process.join()


if __name__ == '__main__':
    main()
