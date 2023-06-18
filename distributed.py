import torch
import numpy as np
from torch import nn
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T
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
import os

# General constants
STATE_DIM = (1, SCREEN_SIZE[0], SCREEN_SIZE[1])

# Training constants
LR = 1e-4
GAMMA = 0.99
BATCH_SIZE = 8
EPOCHS = 100000
TARGET_UPDATE_FREQUENCY = 50
STEPS = 4
DECAY_RATE_CHANGE = 5
MAX_EPSILON = .95

# Memory constants
MEMORY_SIZE = 200
ALPHA = 0.9
BETA = 0.4

possible_actions = [0, 1, 2, 3]


class TrainingProcess:
    def __init__(
            self,
            predict_network,
            target_network,
            optimizer,
            device,
            epoch,
            metrics,
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
        self.data_updates = data_updates
        self.samples = samples
        self.memory_size = memory_size
        self.sample_request_event = sample_request_event
        self.loss = loss
        self.epoch = epoch
        self.terminated = terminated
        self.model_path = model_path
        self.save_path = save_path
        self.optimizer = optimizer

    def training_step(self, gamma, sample):
        self.predict_network.train()
        self.target_network.eval()

        batch, weights, tree_idxs = sample
        states, next_states, actions, rewards, terminated = batch
        states = states.to(torch.float32).to(self.device).detach()
        next_states = next_states.to(torch.float32).to(self.device).detach()
        actions = actions.to(torch.int64).to(self.device).detach()
        rewards = rewards.to(self.device).detach()
        terminated = terminated.to(self.device).detach()
        weights = weights.to(self.device).detach()

        q_values = self.predict_network(states / 255).gather(1, actions)

        with torch.no_grad():
            next_state_q_values = self.target_network(next_states / 255).max(axis=1)[0]
        target_q_values = rewards + (1 - terminated) * gamma * next_state_q_values

        huber = nn.SmoothL1Loss(reduction='none')
        loss = huber(q_values, target_q_values.unsqueeze(1))
        loss = torch.mean(loss * weights)
        td_error = torch.abs(q_values.squeeze(1) - target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.predict_network.parameters(), 100)
        self.optimizer.step()

        return tree_idxs, td_error, loss

    def training_loop(self, epochs, batch_size, gamma):
        while int(self.memory_size.value) < batch_size:
            pass

        self.load_model()

        for epoch in range(epochs):
            self.epoch.value = epoch
            self.sample_request_event.set()
            if self.samples.poll(timeout=999):
                self.sample_request_event.clear()
                sample = self.samples.recv()
                tree_idxs, td_error, loss = self.training_step(gamma, sample)
                del sample

                if epoch % TARGET_UPDATE_FREQUENCY == 0:
                    self.target_network.load_state_dict(self.predict_network.state_dict())
                    print(
                        f'UPDATE [{epoch // TARGET_UPDATE_FREQUENCY}] '
                        f'Loss: {loss} '
                        f'Average rewards: {self.metrics[0]} '
                        f'Average episode length: {self.metrics[1]}')
                    print(
                        f'Highest reward: {self.metrics[2]} '
                        f'Longest episode: {self.metrics[3]}')

                self.data_updates.send({'idxs': tree_idxs, 'priorities': td_error.detach()})
                self.loss.send(loss.detach().item())
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
            action_dim=1,
            alpha=ALPHA,
            beta=BETA
        )
        self.memory.load_experience(destination=self.save_path)

        self.metrics = TrainMatrics()

    def update_priorities(self):
        if self.data_updates.poll():
            update = self.data_updates.recv()
            self.memory.update_priorities(data_idxs=update['idxs'], priorities=update['priorities'])

    def update_data(self):
        if self.data.poll():
            data = self.data.recv()
            self.memory.add(data)
            self.metrics.push_reward(data[3], data[4])

    def send_sample(self):
        if self.sample_request_event.is_set():
            self.samples.send(self.memory.sample(BATCH_SIZE))

    def update_metrics(self):
        if self.loss.poll():
            self.metrics.push_loss(self.loss.recv())

        self.metrics.calculate()
        self.metrics_summary[0] = self.metrics.average_rewards(TARGET_UPDATE_FREQUENCY)
        self.metrics_summary[1] = self.metrics.average_episode_length(TARGET_UPDATE_FREQUENCY)
        self.metrics_summary[2] = self.metrics.highest_reward
        self.metrics_summary[3] = self.metrics.longest_episode

    def save_state(self):
        self.metrics.save(self.save_path, 'metrics')
        self.memory.save_experience(self.save_path)

    def run(self):
        self.init()
        try:
            while True:
                self.memory_size.value = len(self.memory)

                self.update_data()
                self.update_priorities()
                self.send_sample()
                self.update_metrics()

                if self.terminated.value:
                    break

        except (Exception, KeyboardInterrupt) as e:
            traceback.print_exc()
            self.save_state()
        self.save_state()


class DataCollectionProcess(Process):
    def __init__(
            self,
            predict_network,
            data,
            epoch,
            device,
            data_collection_lock,
            terminated
    ):
        Process.__init__(self)
        self.predict_network = predict_network
        self.data = data
        self.terminated = terminated
        self.epoch = epoch
        self.device = device
        self.data_collection_lock = data_collection_lock
        self.game = None

    def init(self):
        self.game = GameEnviroment(size=SCREEN_SIZE, fps=FPS, fps_limit=False, training_mode=True)
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
        state = self._preprocess_state(self.game.get_state()).to(torch.float32).to(self.device)

        action = self.epsilon_greedy_policy(epsilon, state)

        reward, next_state, terminated = self.game.step(int(action))

        reward = torch.tensor(reward)
        terminated = torch.tensor(terminated)
        next_state = self._preprocess_state(next_state)

        data = (state, next_state, action, reward, terminated)
        return data

    def _preprocess_state(self, state):
        with torch.no_grad():
            state = F.to_pil_image(state)
            state = F.to_grayscale(state)
            state = F.resize(state, INPUT_SIZE[1:3], interpolation=T.InterpolationMode.BILINEAR)
            state = F.pil_to_tensor(state)
            state = state.squeeze().reshape((1, *INPUT_SIZE))
        return state

    def _get_epsilon(self, epoch):
        return min(1 - 1 * ((1 - float(epoch) / EPOCHS) ** DECAY_RATE_CHANGE), MAX_EPSILON)

    def collect_data(self, steps):
        for step in range(steps):
            epsilon = self._get_epsilon(self.epoch.value)
            self.data.send(self.do_one_step(epsilon))

    def run(self):
        self.init()
        self.collect_data(200)
        try:
            while True:
                if self.terminated.value:
                    break

                self.collect_data(STEPS)
        except Exception:
            traceback.print_exc()


def main():
    torch.multiprocessing.set_start_method('spawn', force=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predict_network = DQN().to(device)
    target_network = DQN().to(device)
    target_network.load_state_dict(predict_network.state_dict())
    optimizer = torch.optim.AdamW(predict_network.parameters(), lr=LR, amsgrad=True)
    predict_network.share_memory()

    metrics = Array('d', 4)
    data_recv, data_sender = Pipe()
    data_updates_recv, data_updated_sender = Pipe()
    samples_recv, samples_sender = Pipe()
    loss_recv, loss_sender = Pipe()
    sample_request_event = Event()
    data_collection_lock = RLock()
    terminated = Value('b', False)
    epoch = Value('i', 0)
    memory_size = Value('i', 0)

    training_process = TrainingProcess(
        predict_network=predict_network,
        target_network=target_network,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
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
        predict_network=predict_network,
        data=data_sender,
        device=device,
        epoch=epoch,
        data_collection_lock=data_collection_lock,
        terminated=terminated
    ) for _ in range(2)]  # torch.multiprocessing.cpu_count()

    memory_managment_process = MemoryManagmentProcess(
        data=data_recv,
        data_updates=data_updates_recv,
        samples=samples_sender,
        memory_size=memory_size,
        sample_request_event=sample_request_event,
        loss=loss_recv,
        metrics=metrics,
        save_path='backups/data/',
        terminated=terminated
    )

    for data_collection_process in data_collection_processes:
        data_collection_process.start()
    memory_managment_process.start()

    try:
        training_process.training_loop(EPOCHS, BATCH_SIZE, GAMMA)
    except (Exception, KeyboardInterrupt) as e:
        traceback.print_exc()
        training_process.save_model()

    for data_collection_process in data_collection_processes:
        data_collection_process.join()
    memory_managment_process.join()


if __name__ == '__main__':
    main()
