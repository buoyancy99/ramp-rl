import torch
import torch.nn as nn
import wandb
from collections import OrderedDict
import numpy as np
from tqdm import trange
import os


from ..common import BaseAlgorithm
from ..common.utils import OnlineLeastSquare
from .model import ParallelFcQNetwork, ParallelCnnQNetwork


class SfAlgo(BaseAlgorithm):
    def __init__(
        self,
        observation_shape=None,
        action_shape=None,
        arch='512-512',
        num_pi=1,
        basis_dim=256,
        buffer_size=5000,
        argmax_samples=1024,
        reweight=-1,
    ):
        super(SfAlgo, self).__init__(
            observation_shape=observation_shape,
            action_shape=action_shape,
            arch=arch,
            num_pi=num_pi,
            basis_dim=basis_dim,
            buffer_size=buffer_size,
            argmax_samples=argmax_samples,
            reweight=reweight,
        )
        self.is_pixel = (len(observation_shape) == 3)
        self.obs_dim = np.prod(observation_shape).item()
        self.action_dim = np.prod(action_shape).item()
        self.device = torch.device('cuda')

        if self.is_pixel:
            n_channel = self.observation_shape[-1]
            self.qf1s = ParallelCnnQNetwork(
                self.obs_dim, self.action_dim, self.basis_dim, num_pi, n_channel).to(self.device)
            self.qf2s = ParallelCnnQNetwork(
                self.obs_dim, self.action_dim, self.basis_dim, num_pi, n_channel).to(self.device)
        else:
            self.qf1s = ParallelFcQNetwork(self.obs_dim, self.action_dim, self.basis_dim, arch, num_pi).to(self.device)
            self.qf2s = ParallelFcQNetwork(self.obs_dim, self.action_dim, self.basis_dim, arch, num_pi).to(self.device)

        self.valid_pi_indices = None
        self.buffer = None
        self.buffer_top = 0
        self.buffer_full = False
        self.exploration_steps = max(int(self.basis_dim * 1.2), 2048)
        self.online_lsq = OnlineLeastSquare()
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = [np.ones((self.buffer_size, self.basis_dim + 1), dtype=np.float32),
                       np.zeros(self.buffer_size, dtype=np.float32)]
        self.buffer_top = 0
        self.buffer_full = False

    def _predict(self, obs, action_samples):
        batch_size, num_samples, action_dim = action_samples.shape
        obs = torch.from_numpy(obs).to(self.device)
        action_samples = torch.from_numpy(action_samples).to(self.device)
        obs = obs.repeat(num_samples, *[1 for _ in range(len(self.observation_shape))])
        action_samples = action_samples.reshape(batch_size * num_samples, action_dim)

        with torch.no_grad():
            # (batch * num_samples, num_pi, basis_dim)
            psi_hat = torch.min(self.qf1s(obs, action_samples), self.qf2s(obs, action_samples))
            psi_hat = psi_hat.reshape(batch_size, num_samples, self.num_pi, self.basis_dim)
            w = torch.from_numpy(self.calc_w()).to(self.device)
            q_hat = torch.sum(psi_hat * w[None, None, None, :-1], -1) + w[None, None, None, -1]
            q_hat = q_hat.detach().cpu().numpy()

        pi_indices = np.ones(q_hat.shape[:2] + (1, ), dtype=int) * self.valid_pi_indices[:, None, :]
        q_hat = np.take_along_axis(q_hat, pi_indices, axis=2)   # (batch_size, num_samples, num_goals//2)
        q_hat = np.max(q_hat, axis=2).reshape(batch_size, num_samples)
        best_indices = np.argmax(q_hat, axis=1) + np.arange(batch_size) * num_samples
        best_actions = action_samples[best_indices].detach().cpu().numpy()

        return best_actions

    def calc_w(self):
        if not self.online_lsq.initialized:
            return np.ones(self.basis_dim + 1) / self.basis_dim
        w = self.online_lsq.calculate()

        return w

    def _add_to_buffer(self, r_basis, r_gt):
        batch_size = len(r_gt)
        assert batch_size == 1
        space_left = self.buffer_size - self.buffer_top
        if batch_size > space_left:
            self.buffer[0][space_left:] = self.buffer[0][:self.buffer_top]
            self.buffer[1][space_left:] = self.buffer[1][:self.buffer_top]
            self.buffer_top = 0
            self.buffer_full = True

        self.buffer[0][self.buffer_top:self.buffer_top + batch_size, :-1] = r_basis
        self.buffer[1][self.buffer_top:self.buffer_top + batch_size] = r_gt

        if batch_size <= space_left:
            self.buffer_top += batch_size

        if self.buffer_top >= self.exploration_steps or self.online_lsq.initialized:
            if not self.online_lsq.initialized:
                A, b = self.buffer[0][:self.buffer_top], self.buffer[1][:self.buffer_top]
                scale = 1 / np.sqrt(self.reweight + b ** 2) if self.reweight > 0 else np.ones_like(b)
                self.online_lsq.initialize(A * scale[:, None], b * scale)
            else:
                scale = 1 / np.sqrt(self.reweight + r_gt ** 2) if self.reweight > 0 else np.ones_like(r_gt)
                r_basis = np.concatenate([r_basis, np.ones((batch_size, 1))], -1)
                self.online_lsq.update(r_basis * scale[:, None], r_gt * scale)

    def train_online(self, env, num_episodes, projection_net=None, policy_goals=None, relabel_frac=0.5):
        self._calc_valid_pi_indices(env.get_attr('goal'), policy_goals, relabel_frac)
        wandb.define_metric("online/step")
        wandb.define_metric("online/*", step_metric="online/step")
        projection_net = projection_net.to(self.device)
        num_envs = env.num_envs
        return_stat = np.zeros((num_episodes, num_envs))
        success_stat = np.zeros((num_episodes, num_envs))
        step = 0
        for episode_idx in trange(num_episodes, desc='Evaluation Episode',
                                  disable=os.environ.get("DISABLE_TQDM", False)):
            done = False
            obs = env.reset()
            while not done:
                step += 1

                if self.buffer_top >= self.exploration_steps or self.online_lsq.initialized:
                    action_samples = np.stack(
                        [env.action_space.sample() for _ in range(num_envs * self.argmax_samples)], 0)
                    action_samples = action_samples.reshape(num_envs, self.argmax_samples, -1)
                    action = self._predict(obs, action_samples)
                else:
                    action = np.stack([env.action_space.sample() for _ in range(num_envs)])

                obs, reward, done, info = env.step(action)

                reward_basis = projection_net(
                    torch.from_numpy(obs).to(self.device),
                    torch.from_numpy(action).to(self.device),
                ).detach().cpu().numpy()

                self._add_to_buffer(reward_basis, reward)

                done = np.any(done)
                return_stat[episode_idx] += np.array(reward)
                if done:
                    obs = env.reset()

            success_stat[episode_idx] = np.array([item['success'] for item in info])

            wandb.log({'online/return': return_stat[episode_idx].mean().item(),
                       'online/success': success_stat[episode_idx].mean().item(),
                       'online/step': step
                       })

        wandb.run.summary["return_mean"] = return_stat.mean()
        wandb.run.summary["return_max"] = return_stat.max()
        wandb.run.summary["return_std"] = return_stat.std()
        wandb.run.summary["success_rate"] = success_stat.mean()

        return np.mean(return_stat)

    def ensemble_state_dict(self, state_dicts):
        num_pi = len(state_dicts)
        # assert len(state_dicts) == self.num_pi
        new_dict = OrderedDict()
        for k in state_dicts[0].keys():
            new_dict[k] = []

        for i, state_dict in enumerate(state_dicts):
            for k, v in state_dict.items():
                new_dict[k].append(v)

        for k, v in new_dict.items():
            if 'weight' in k:
                out_dim, in_dim = v[0].shape
                new_dict[k] = torch.stack(v).reshape(num_pi * out_dim, in_dim, 1)
            elif 'bias' in k:
                out_dim, = v[0].shape
                new_dict[k] = torch.stack(v).reshape(num_pi * out_dim)

        return new_dict

    @property
    def _state_attributes(self):
        return ['qf1s', 'qf2s']

    def _calc_valid_pi_indices(self, env_goals, policy_goals, relabel_frac):
        assert relabel_frac >= 0.0 and relabel_frac <= 1.0
        num_goals = len(policy_goals)
        num_envs = len(env_goals)
        if policy_goals is None:
            self.valid_pi_indices = np.arange(self.num_pi)[None] * np.ones((num_envs, self.num_pi))
        else:
            env_goals = np.array(env_goals)
            policy_goals = np.stack(policy_goals)
            goal_dists = np.linalg.norm(env_goals[:, None] - policy_goals[None], axis=2)
            self.valid_pi_indices = np.argsort(-goal_dists, axis=1)[:, :int(num_goals * (1 - relabel_frac))]

