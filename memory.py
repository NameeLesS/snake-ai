import random
import torch
import numpy as np
import os
import pickle


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim, alpha=0.9, beta=0.4, eps=1e-2):
        self.tree = SumTree(buffer_size)

        self.alpha = alpha
        self.beta = beta
        self.max_prority = eps
        self.eps = eps

        self.states = torch.empty(buffer_size, *state_dim, dtype=torch.uint8)
        self.next_states = torch.empty(buffer_size, *state_dim, dtype=torch.uint8)
        self.rewards = torch.empty(buffer_size, dtype=torch.uint8)
        self.actions = torch.empty(buffer_size, action_dim, dtype=torch.uint8)
        self.terminated = torch.empty(buffer_size, dtype=torch.uint8)

        self.buffer_size = buffer_size
        self.size = 0
        self.write_idx = 0

    def add(self, transition):
        state, next_state, action, reward, terminated = transition
        state, next_state, action, reward, terminated = state.detach(), next_state.detach(), action.detach(), reward.detach(), terminated.detach()

        self.tree.add(self.max_prority, self.write_idx)

        self.states[self.write_idx] = state
        self.next_states[self.write_idx] = next_state
        self.rewards[self.write_idx] = reward
        self.actions[self.write_idx] = action
        self.terminated[self.write_idx] = terminated

        self.write_idx += 1
        if self.write_idx == self.buffer_size:
            self.write_idx = 0

        if self.size < self.buffer_size:
            self.size += 1

    def sample(self, batch_size):
        if batch_size > self.size:
            return

        tree_idxs = []
        data_idxs = []
        priorities = torch.empty(batch_size, 1)

        segment_size = self.tree.sumcum / batch_size
        for i in range(batch_size):
            lower_bound, upper_bound = i * segment_size, (i + 1) * segment_size

            tree_idx, priority, data_idx = self.tree.get(random.uniform(lower_bound, upper_bound))
            tree_idxs.append(tree_idx)
            data_idxs.append(data_idx)
            priorities[i] = priority

        probs = priorities / self.tree.sumcum

        weights = ((1 / self.size) * (1 / probs)) ** self.beta
        weights = weights / weights.max()

        batch = (
            self.states[data_idxs],
            self.next_states[data_idxs],
            self.actions[data_idxs],
            self.rewards[data_idxs],
            self.terminated[data_idxs]
        )

        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(idx, priority)
            self.max_prority = max(self.max_prority, priority)

    def save_experience(self, destination=''):
        if not os.path.exists(destination):
            os.makedirs(destination)

        with open(os.path.join(destination, 'data.pt'), 'wb+') as f:
            torch.save({
                'states': self.states,
                'rewards': self.rewards,
                'next_states': self.next_states,
                'actions': self.actions,
                'terminated': self.terminated,
                'size': self.size
            }, f)

        with open(os.path.join(destination, 'meta.pkl'), 'wb+') as f:
            pickle.dump({
                'nodes': self.tree.nodes,
                'data': self.tree.data,
                'size': self.tree.size,
                'capacity': self.tree.capacity
            }, f)

    def load_experience(self, destination='', data_file='data.pt', meta_file='meta.pkl'):
        if (
                not os.path.exists(os.path.join(destination, data_file)) or
                not os.path.exists(os.path.join(destination, meta_file))
        ):
            print('Data not found')
            return

        with open(os.path.join(destination, data_file), 'rb+') as f:
            data = torch.load(f)

            # cut the data which is redundant
            size = data['size']
            if size > self.buffer_size:
                size = self.buffer_size

            # set write index to write new data at the end of the buffer
            if size < self.buffer_size:
                self.write_idx = size

            self.states[:size] = data['states'][:size]
            self.next_states[:size] = data['next_states'][:size]
            self.rewards[:size] = data['rewards'][:size]
            self.actions[:size] = data['actions'][:size]
            self.terminated[:size] = data['terminated'][:size]
            self.size = size

        with open(os.path.join(destination, meta_file), 'rb+') as f:
            meta = pickle.load(f)
            size = meta['size']
            capacity = meta['capacity']

            # cut the data which is redundant
            if size > self.buffer_size:
                size = self.buffer_size

            if capacity > self.buffer_size:
                capacity = self.buffer_size

            # set write index to write new data at the end of the tree
            if size < self.buffer_size:
                self.tree.last_idx = size

            self.tree.nodes[:capacity * 2 - 1] = meta['nodes'][:capacity * 2 - 1]
            self.tree.data[:size] = meta['data'][:size]

    def __len__(self):
        return self.size


class SumTree:
    def __init__(self, capacity):
        self.nodes = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=int)
        self.capacity = capacity
        self.last_idx = 0
        self.size = 0

    def add(self, prority, data):
        idx = self.last_idx + self.capacity - 1
        self.data[self.last_idx] = data

        self.update(idx, prority)

        self.last_idx += 1
        if self.last_idx >= self.capacity:
            self.last_idx = 0

        if self.size < self.capacity:
            self.size += 1

    def update(self, idx, prority):
        change = prority - self.nodes[idx]
        self.nodes[idx] = prority

        self._propagate(idx, change)

    def get(self, sumcum):
        if sumcum >= self.sumcum:
            return None

        idx = self._retrieve(0, sumcum)
        data_idx = idx - self.capacity + 1
        return idx, self.nodes[idx], self.data[data_idx]

    def _propagate(self, idx, change):
        parent = self.parent(idx)
        self.nodes[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, sumcum):
        left = self.left_child(idx)
        right = self.right_child(idx)

        if left >= 2 * self.capacity - 1:
            return idx

        if self.nodes[left] >= sumcum:
            return self._retrieve(left, sumcum)
        else:
            return self._retrieve(right, sumcum - self.nodes[left])

    @staticmethod
    def parent(idx):
        return (idx - 1) // 2

    @staticmethod
    def right_child(idx):
        return idx * 2 + 2

    @staticmethod
    def left_child(idx):
        return idx * 2 + 1

    @property
    def sumcum(self):
        return self.nodes[0]

    def __len__(self):
        return self.size
