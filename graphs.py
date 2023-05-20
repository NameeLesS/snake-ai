import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import os


class Graphs:
    def __init__(self):
        self.rewards = np.array([[0, 0]], dtype=np.int)
        self.reward_series = []
        self.losses = np.array([], dtype=np.float32)

    def plot_rewards(self):
        plt.plot(self.reward_series)
        plt.xlabel('Series number')
        plt.ylabel('Reward')

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    def get_series_reward(self):
        if self.rewards[self.rewards[:, 1] == 1].any():
            terminated_idx = int(np.where(self.rewards[:, 1] == 1)[0]) + 1
            reward_series = self.rewards[:terminated_idx]
            self.rewards = self.rewards[terminated_idx:]
            self.reward_series.append({'Episodes': len(reward_series), 'Rewards': np.sum(reward_series, 0)[0]})

            return self.reward_series
        else:
            return (len(self.rewards), np.sum(self.rewards, 0)[0], self.reward_series[-1] if len(self.reward_series) else None)

    def push_loss(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().numpy()
        self.losses = np.append(self.losses, loss)

    def push_reward(self, reward, terminated):
        if isinstance(reward, torch.Tensor):
            reward = reward.detach().cpu().numpy()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.detach().cpu().numpy()
        self.rewards = np.vstack((self.rewards, np.array([reward, terminated])))

    def save(self, path, name_reward, name_loss):
        rewards_path = os.path.join(path, '{name_rewards}.pkl')
        losses_path = os.path.join(path, '{name_loss}.pkl')

        with open(rewards_path, 'wb+') as f:
            pickle.dump(self.reward_series, f)

        with open(losses_path, 'wb+') as f:
            pickle.dump(self.losses, f)
