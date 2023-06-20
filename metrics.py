import numpy as np
import torch
import pickle
import os


class TrainMatrics:
    def __init__(self):
        self.rewards = {}
        self.losses = np.array([], dtype=np.float32)
        self._episode_lengths = []
        self._episode_rewards = []

    def push_loss(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().numpy()
        self.losses = np.append(self.losses, loss)

    def push_reward(self, game_id, reward, terminated):
        if isinstance(reward, torch.Tensor) or isinstance(terminated, torch.Tensor):
            reward = reward.detach().numpy()
            terminated = terminated.detach().numpy()
        if game_id in self.rewards:
            self.rewards[game_id] = np.vstack((self.rewards[game_id], np.array([reward, terminated])))
        else:
            self.rewards[game_id] = np.array([[reward, terminated]])

    def calculate(self):
        for game_id, rewards in self.rewards.items():
            if rewards[rewards[:, 1] == 1].any():
                terminated_idxs = np.where(rewards[:, 1] == 1)[0] + 1
                reward_series = np.split(rewards, terminated_idxs)[:-1]
                self.rewards[game_id] = self.rewards[game_id][terminated_idxs[-1]:]
                self._episode_lengths.extend(list(map(lambda series: len(series), reward_series)))
                self._episode_rewards.extend(list(map(lambda series: np.sum(series, 0)[0], reward_series)))

    def save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)

        save_path = os.path.join(path, f'{name}.pkl')
        data = {
            'rewards': self._episode_rewards,
            'episodes': self._episode_lengths,
            'losses': self.losses,
        }

        with open(save_path, 'wb+') as f:
            pickle.dump(data, f)

    def average_rewards(self, length):
        episodes = self._episode_rewards[-length:]
        return sum(episodes) / len(episodes) if len(self._episode_rewards) else 0

    def average_episode_length(self, length):
        episodes = self._episode_lengths[-length:]
        return sum(episodes) / len(episodes) if len(self.episode_lengths) else 0

    @property
    def loss(self):
        return self.losses[-1] if len(self.losses) else 0

    @property
    def episode_lengths(self):
        return self._episode_lengths

    @property
    def episode_rewards(self):
        return self._episode_rewards

    @property
    def highest_reward(self):
        return max(self._episode_rewards) if len(self._episode_rewards) else 0

    @property
    def longest_episode(self):
        return max(self._episode_lengths) if len(self._episode_lengths) else 0
