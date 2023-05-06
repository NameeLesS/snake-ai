import matplotlib.pyplot as plt
import numpy as np


class Graphs:
    def __init__(self):
        self.rewards = np.array([[0, 0]], dtype=np.int)
        self.losses = np.array([], dtype=np.float32)

    def plot_rewards(self):
        reward_series = self.get_series_reward()
        plt.plot(reward_series)
        plt.xlabel('Series number')
        plt.ylabel('Reward')

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    def get_series_reward(self):
        terminated_indices = np.array(np.where(self.rewards[:, 1] == 1)).reshape(-1) + 1
        reward_series = np.array_split(self.rewards, terminated_indices)
        reward_series = list(map(lambda rewards: rewards.sum(), reward_series))
        return reward_series

    def push_loss(self, loss):
        self.losses = np.append(self.losses, loss)

    def push_reward(self, reward, terminated):
        self.rewards = np.vstack((self.rewards, np.array([reward, terminated])))

