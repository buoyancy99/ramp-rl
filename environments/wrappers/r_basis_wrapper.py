import gym
import torch
import torch.nn as nn
import numpy as np


class MlpNet(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim=512):
        super(MlpNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, out_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class RandMlpProjNet(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim=32, last_relu=False):
        super(RandMlpProjNet, self).__init__()
        layers = [
            nn.Conv1d(in_dim, h_dim * out_dim, 1),
            nn.ReLU(),
            nn.Conv1d(h_dim * out_dim, h_dim * out_dim, 1, groups=out_dim),
            nn.ReLU(),
            nn.Conv1d(h_dim * out_dim, out_dim, 1, groups=out_dim)
        ]
        if last_relu:
            layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*layers)

    def forward(self, obs, action):
        assert obs.dim() == 2
        x = torch.cat([obs, action], 1)
        return self.encoder(x[:, :, None])[:, :, 0]


class RandCnnNet(nn.Module):
    def __init__(self, action_dim, out_dim, h_dim=16, last_relu=False, channel=3):
        super().__init__()
        self.action_dim = action_dim
        group = 8
        self.cnn = nn.Sequential(
            nn.Conv2d(channel, h_dim * group, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(h_dim * group, h_dim * 2 * group, kernel_size=4, stride=2, groups=group),
            nn.ReLU(),
            nn.Conv2d(h_dim * 2 * group, h_dim * group, kernel_size=3, stride=1, groups=group)
        )
        self.action_mlp = nn.Linear(action_dim, 2048)
        layers = [
            nn.ReLU(),
            nn.Conv1d(h_dim * group * 7 * 7 + 2048, h_dim * out_dim, 1),
            nn.ReLU(),
            nn.Conv1d(h_dim * out_dim, h_dim * out_dim, 1, groups=out_dim),
            nn.ReLU(),
            nn.Conv1d(h_dim * out_dim, out_dim, 1, groups=out_dim),
        ]
        if last_relu:
            layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs, action):
        assert obs.dim() == 4
        obs = obs.permute(0, 3, 1, 2).contiguous()
        img_feature = self.cnn(obs)
        action_feature = self.action_mlp(action)
        x = torch.cat([img_feature.flatten(1), action_feature], 1)
        return self.mlp(x[:, :, None])[:, :, 0]


class RewardBasisWrapper(gym.Wrapper):
    def __init__(self, env, n_rewards, include_action):
        super(RewardBasisWrapper, self).__init__(env)
        self.n_rewards = n_rewards
        self.include_action = include_action
        self.in_dim = np.prod(self.env.observation_space.shape) + np.prod(self.env.action_space.shape) * include_action
        self._last_obs = None

    def compute_reward_basis(self, action, reward):
        raise NotImplementedError

    def step(self, action):
        obs, original_reward, done, info = self.env.step(action)
        reward = self.compute_reward_basis(action, original_reward)
        self._last_obs = obs
        info['original_reward'] = original_reward
        return obs, reward, done, info

    def reset(self):
        self._last_obs = self.env.reset()
        return self._last_obs


class RandomRewardWrapper(RewardBasisWrapper):
    def __init__(self, env, n_rewards, include_action=True, device='cpu', seed=0):
        super(RandomRewardWrapper, self).__init__(env, n_rewards, include_action)
        self.device = torch.device(device)
        self.is_pixel = len(self.env.observation_space.shape) == 3
        if self.is_pixel:
            self.n_channels = self.env.observation_space.shape[2]
        self.net = self._build_mlp(seed)

    def _build_mlp(self, seed):
        torch.manual_seed(seed)
        if self.is_pixel:
            net = RandCnnNet(
                np.prod(self.env.action_space.shape) * self.include_action, 
                self.n_rewards, 
                channel=self.n_channels,
            )
        else:
            net = RandMlpProjNet(self.in_dim, self.n_rewards)
        return net.to(self.device)

    def compute_reward_basis(self, action, reward):
        obs = self._last_obs / 255.0 * 2 - 1.0 if self.is_pixel else self._last_obs
        obs = torch.from_numpy(obs.astype(np.float32)).to(self.device)[None]
        action = torch.from_numpy(action).to(self.device)[None]
        with torch.no_grad():
            x = self.net(obs, action)[0]

        return x.detach().cpu().numpy()


class LearnedRewardWrapper(RandomRewardWrapper):
    def __init__(self, env, n_rewards, include_action=True, device='cpu', state_dict=None):
        super(LearnedRewardWrapper, self).__init__(env, n_rewards, include_action, device)
        if state_dict is not None:
            self.net.load_state_dict(state_dict)

    def _build_mlp(self, seed):
        torch.manual_seed(seed)
        if self.is_pixel:
            net = RandCnnNet(np.prod(self.env.action_space.shape) * self.include_action,
                             self.n_rewards, last_relu=True)
        else:
            net = RandMlpProjNet(self.in_dim, self.n_rewards, last_relu=True)
        return net.to(self.device)


class PolynomialRewardWrapper(RewardBasisWrapper):
    def __init__(self, env, include_action=True):
        super(PolynomialRewardWrapper, self).__init__(env, None, include_action)
        self.n_rewards = self._compute_n_reward()

    def _compute_n_reward(self):
        return int((self.in_dim + 2) * (self.in_dim + 1) / 2) - 1
    
    def compute_reward_basis(self, action, reward):
        x = np.ones(self.in_dim + 1)
        x[1:] = np.concatenate([self._last_obs.flatten(), action.flatten()])
        x = x[None] * x[:, None]
        triu_inds = np.triu_indices(self.in_dim + 1)
        return x[triu_inds[0][1:], triu_inds[1][1:]]
    
    def projection_function(self, obs, action):
        x = torch.ones((len(obs), self.in_dim + 1))
        x[:, 1:] = torch.cat([obs, action], -1)
        x = x[:, None, :] * x[:, :, None]
        triu_inds = torch.triu_indices(self.in_dim + 1, self.in_dim + 1)
        return x[:, triu_inds[0, 1:], triu_inds[1, 1:]]


class DummyRewardWrapper(RewardBasisWrapper):
    def __init__(self, env):
        super(DummyRewardWrapper, self).__init__(env, None, False)
        self.n_rewards = 1

    def compute_reward_basis(self, action, reward):
        return np.array([reward])
