import numpy as np

from .dataset import BufferTrajSliceDataset


class RampRolloutBuffer:
    def __init__(self, buffer_size, observation_shape, action_shape, basis_dim, episode_len = None):
        self.buffer_size = buffer_size
        self.observation_shape = tuple(observation_shape)
        self.action_shape = tuple(action_shape)
        self.basis_dim = basis_dim + 1

        self.top = 0
        self.last_top = 0
        self.total = 0
        self.actions = np.zeros((self.buffer_size, ) + self.action_shape)
        self.observations = np.zeros((self.buffer_size,) + self.observation_shape)
        self.rewards = np.zeros((self.buffer_size, ))
        self.phi_bases = np.ones((self.buffer_size, self.basis_dim))
        self.dones = np.zeros((self.buffer_size, ))
        self.episode_len = episode_len

    def add(self, obs, action, reward, done, phi):
        self.top = self.top % self.buffer_size
        self.observations[self.top] = obs
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.phi_bases[self.top, :-1] = phi
        self.dones[self.top] = done
        self.top += 1
        self.total += 1

        if done:
            if self.episode_len is None:
                self.episode_len = self.top
            else:
                assert not self.buffer_size % self.episode_len
                assert self.episode_len == self.top - self.last_top
            self.last_top = self.top % self.buffer_size

    def reset(self):
        self.top = 0
        self.last_top = 0

    def regression_data(self):
        return self.phi_bases[:self.top], self.rewards[:self.top]

    def get_rollout(self, obs, action, reward):
        assert self.last_top == self.top % self.buffer_size

        rollout = {
            'obs': obs.reshape((-1, self.episode_len) + self.observation_shape),
            'action': action.reshape((-1, self.episode_len) + self.action_shape),
            'reward': reward.reshape((-1, self.episode_len)),
        }

        return rollout

    def get_dataset(self, horizon, n_episodes = None):
        assert self.last_top == self.top % self.buffer_size
        if not n_episodes:
            assert not self.buffer_size % self.episode_len
            n_episodes = self.buffer_size // self.episode_len

        total_len = n_episodes * self.episode_len
        if self.total <= self.buffer_size:
            start_idx = max(0, self.top - total_len)
            obs = self.observations[start_idx:self.top]
            action = self.actions[start_idx:self.top]
            reward = self.rewards[start_idx:self.top]
        else:
            start_idx = self.top - total_len
            assert total_len <= self.buffer_size
            if self.top >= total_len:
                obs = self.observations[start_idx:self.top]
                action = self.actions[start_idx:self.top]
                reward = self.rewards[start_idx:self.top]
            else:
                obs = np.concatenate([self.observations[self.top - total_len:], self.observations[:self.top]])
                action = np.concatenate([self.actions[self.top - total_len:], self.actions[:self.top]])
                reward = np.concatenate([self.rewards[self.top - total_len:], self.rewards[:self.top]])

        rollout = {
            'obs': obs.reshape((-1, self.episode_len) + self.observation_shape),
            'action': action.reshape((-1, self.episode_len) + self.action_shape),
            'reward': reward.reshape((-1, self.episode_len)),
        }

        return BufferTrajSliceDataset(rollout, horizon=horizon)
