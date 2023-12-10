import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform


def extend_and_repeat(tensor, dim, repeat):
    # Extend and repeast the tensor along dim axie and repeat it
    ones_shape = [1 for _ in range(tensor.ndim + 1)]
    ones_shape[dim] = repeat
    return torch.unsqueeze(tensor, dim) * tensor.new_ones(ones_shape)


def soft_target_update(network, target_network, soft_target_update_rate):
    target_network_params = {k: v for k, v in target_network.named_parameters()}
    for k, v in network.named_parameters():
        target_network_params[k].data = (
            (1 - soft_target_update_rate) * target_network_params[k].data
            + soft_target_update_rate * v.data
        )


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and (observations.ndim == 2 or observations.ndim == 4):
            multiple_actions = True
            rep = actions.shape[1]
            observations = extend_and_repeat(observations, 1, rep).reshape(-1, *observations.shape[1:])
            if 'goals' in kwargs.keys() and kwargs['goals'] is not None:
                kwargs['goals'] = \
                    extend_and_repeat(kwargs['goals'], 1, rep).reshape(-1, kwargs['goals'].shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])

        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, rep, -1)
        return q_values
    return wrapped


class FullyConnectedNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, arch='128-128', orthogonal_init=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split('-')]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            if orthogonal_init:
                nn.init.orthogonal_(fc.weight, gain=np.sqrt(2))
                nn.init.constant_(fc.bias, 0.0)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)
        if orthogonal_init:
            nn.init.orthogonal_(last_fc.weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(last_fc.weight, gain=1e-2)

        nn.init.constant_(last_fc.bias, 0.0)
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)


class CnnFeatureNetwork(nn.Module):
    def __init__(self, flat_dim, output_dim, n_channel):
        super().__init__()
        self.flat_dim = flat_dim
        self.output_dim = output_dim
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(n_channel, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
        )

        if self.flat_dim:
            flat_feature_dim = 128
            self.mlp_encoder = nn.Linear(flat_dim, flat_feature_dim)
        else:
            flat_feature_dim = 0
            self.mlp_encoder = None

        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Linear((64 * 7 * 7 + flat_feature_dim), 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, pixel_input, flat_input):
        if pixel_input.ndim == 5:
            batch_size, rep, c, h, w = pixel_input.shape
            pixel_input = pixel_input.view(batch_size * rep, c, h, w)
            pixel_features = self.cnn_encoder(pixel_input).view(batch_size, rep, 64 * 7 * 7)
        else:
            pixel_features = self.cnn_encoder(pixel_input).flatten(1)

        if self.flat_dim:
            flat_feature = self.mlp_encoder(flat_input)
            input_tensor = torch.cat([pixel_features, flat_feature], -1)
        else:
            input_tensor = pixel_features
        return self.network(input_tensor)


class ReparameterizedTanhGaussian(nn.Module):

    def __init__(self, log_std_min=-20.0, log_std_max=2.0, no_tanh=False):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.no_tanh = no_tanh

    def log_prob(self, mean, log_std, sample):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )
        return torch.sum(action_distribution.log_prob(sample), dim=-1)

    def forward(self, mean, log_std, deterministic=False):
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if self.no_tanh:
            action_distribution = Normal(mean, std)
        else:
            action_distribution = TransformedDistribution(
                Normal(mean, std), TanhTransform(cache_size=1)
            )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(
            action_distribution.log_prob(action_sample), dim=-1
        )

        return action_sample, log_prob


class TanhGaussianPolicy(nn.Module):
    def __init__(self, observation_dim, action_dim, goal_dim=0, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0,
                 orthogonal_init=False, no_tanh=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = FullyConnectedNetwork(
            observation_dim, 2 * action_dim, arch, orthogonal_init
        )
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions, goals):
        if self.goal_dim:
            observations = torch.cat([observations, goals], -1)
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, goals, deterministic=False, repeat=None):
        if self.goal_dim:
            observations = torch.cat([observations, goals], -1)
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian(mean, log_std, deterministic)


class CnnTanhGaussianPolicy(nn.Module):
    def __init__(self, goal_dim, action_dim, arch='256-256',
                 log_std_multiplier=1.0, log_std_offset=-1.0,
                 orthogonal_init=False, no_tanh=False, n_channel=3):
        super().__init__()
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.no_tanh = no_tanh

        self.base_network = CnnFeatureNetwork(goal_dim, 2 * action_dim, n_channel)
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.tanh_gaussian = ReparameterizedTanhGaussian(no_tanh=no_tanh)

    def log_prob(self, observations, actions, goals=None):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
            if self.goal_dim:
                goals = extend_and_repeat(goals, 1, actions.shape[1])
        base_network_output = self.base_network(observations, goals)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian.log_prob(mean, log_std, actions)

    def forward(self, observations, goals=None, deterministic=False, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
            if self.goal_dim:
                goals = extend_and_repeat(goals, 1, repeat)
        base_network_output = self.base_network(observations, goals)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        return self.tanh_gaussian(mean, log_std, deterministic)


class SamplerPolicy(object):
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def __call__(self, observations, deterministic=False):
        with torch.no_grad():
            observations = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
            actions, _ = self.policy(observations, deterministic)
            actions = actions.cpu().numpy()
        return actions


class FullyConnectedQFunction(nn.Module):
    def __init__(self, observation_dim, action_dim, basis_dim, goal_dim=0, arch='256-256', orthogonal_init=False):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.basis_dim = basis_dim
        self.goal_dim = goal_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.network = FullyConnectedNetwork(
            observation_dim + action_dim + goal_dim, basis_dim, arch, orthogonal_init
        )

    @multiple_action_q_function
    def forward(self, observations, actions, goals=None):
        if self.goal_dim:
            observations = torch.cat([observations, goals], -1)
        input_tensor = torch.cat([observations, actions], dim=-1)
        return self.network(input_tensor)


class CnnQFunction(nn.Module):
    def __init__(self, basis_dim, goal_dim, action_dim, n_channel=3):
        super().__init__()
        self.basis_dim = basis_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.network = CnnFeatureNetwork(goal_dim + action_dim, basis_dim, n_channel)

    @multiple_action_q_function
    def forward(self, observations, actions, goals=None):
        if self.goal_dim:
            actions = torch.cat([actions, goals], dim=-1)
        return self.network(observations, actions)


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32)
        )

    def forward(self):
        return self.constant


class ParallelFcQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, basis_dim, arch='256-256', num_q=1):
        super(ParallelFcQNetwork, self).__init__()
        self.basis_dim = basis_dim
        self.num_q = num_q
        input_dim = observation_dim + action_dim

        d = input_dim
        modules = []
        hidden_sizes = [int(h) for h in arch.split('-')]

        for hidden_size in hidden_sizes:
            fc = nn.Conv1d(d * num_q, hidden_size * num_q, 1, groups=num_q)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        modules.append(nn.Conv1d(d * num_q, basis_dim * num_q, 1, groups=num_q))

        self.network = nn.Sequential(*modules)

    def forward(self, obs, action):
        batch_size = obs.shape[0]
        obs = obs.flatten(1)
        action = action.flatten(1)
        batch_input = torch.cat([obs, action], 1).repeat(1, self.num_q)
        result = self.network(batch_input[:, :, None])[:, :, 0]
        return result.view(batch_size, self.num_q, self.basis_dim)


class ParallelCnnQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, basis_dim, num_q=1, n_channel=3):
        super(ParallelCnnQNetwork, self).__init__()
        self.basis_dim = basis_dim
        self.num_q = num_q

        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(n_channel * num_q, 32 * num_q, kernel_size=8, stride=4, groups=num_q),
            nn.ReLU(),
            nn.Conv2d(32 * num_q, 64 * num_q, kernel_size=4, stride=2, groups=num_q),
            nn.ReLU(),
            nn.Conv2d(64 * num_q, 64 * num_q, kernel_size=3, stride=1, groups=num_q)
        )

        self.network = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d((64 * 7 * 7 + action_dim) * num_q, 256 * num_q, 1, groups=num_q),
            nn.ReLU(),
            nn.Conv1d(256 * num_q, basis_dim * num_q, 1, groups=num_q),
        )

    def forward(self, obs, action):
        batch_size = obs.shape[0]
        obs = obs.repeat(1, self.num_q, 1, 1, 1)
        action = action.repeat(1, self.num_q)[:, :, None]
        result = self.network(obs, action)[:, :, 0]
        return result.view(batch_size, self.num_q, self.basis_dim)
