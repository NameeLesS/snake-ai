import numpy as np
import torch
import pickle
import os


class TrainMatrics:
    def __init__(self):
        self.rewards = np.array([[0, 0]], dtype=np.int)
        self.losses = np.array([], dtype=np.float32)
        self._episode_lengths = []
        self._episode_rewards = []

    def push_loss(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().numpy()
        self.losses = np.append(self.losses, loss)

    def push_reward(self, reward, terminated):
        if isinstance(reward, torch.Tensor) or isinstance(terminated, torch.Tensor):
            reward = reward.detach().numpy()
            terminated = terminated.detach().numpy()
        self.rewards = np.vstack((self.rewards, np.array([reward, terminated])))

    def calculate(self):
        if self.rewards[self.rewards[:, 1] == 1].any():
            terminated_idx = int(np.where(self.rewards[:, 1] == 1)[0]) + 1
            reward_series = self.rewards[:terminated_idx]
            self.rewards = self.rewards[terminated_idx:]
            self._episode_lengths.append(len(reward_series))
            self._episode_rewards.append(np.sum(reward_series, 0)[0])

    def save(self, path, name):
        save_path = os.path.join(path, f'{name}.pkl')
        data = {
            'rewards': self._episode_rewards,
            'episodes': self._episode_lengths,
            'losses': self.losses,
        }

        with open(save_path, 'wb+') as f:
            pickle.dump(data, f)

    @property
    def average_rewards(self):
        return sum(self._episode_rewards) / len(self._episode_rewards) if len(self._episode_rewards) else None

    @property
    def loss(self):
        return self.losses[-1] if len(self.losses) else None

    @property
    def average_episode_length(self):
        return sum(self._episode_lengths) / len(self._episode_lengths) if len(self.episode_lengths) else None

    @property
    def episode_lengths(self):
        return self._episode_lengths

    @property
    def episode_rewards(self):
        return self._episode_rewards


