import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm
import numpy as np
import os
import wandb

from torch.distributions import Normal
from torch.utils.data import DataLoader, random_split
from sklearn.linear_model import LinearRegression
from stable_baselines3.common.utils import polyak_update

from ..common import BaseAlgorithm
from .models import LstmBasisNet, MlpBasisNet, CnnBasisNet, MlpVfNet, CnnVfNet
from .dataset import TrajSliceDataset
from ..common.utils import OnlineLeastSquare
from .rollout_buffer import RampRolloutBuffer


class MPCAlgo(BaseAlgorithm):
    def __init__(
        self,
        observation_shape=None,
        action_shape=None,
        basis_dim=1,
        learning_rate=3e-4,
        num_epoch=1,
        batch_size=128,
        mpc_samples=32,
        mpc_horizon=12,
        buffer_size=12800,
        gamma=0.9,
        reweight=0.5,
        ensemble=8,
        disagreement_coef=1.0,
        normalize_basis=False,
        successs_update_rate=0.01,
        planner="random_shooting",
        cem_iters=10,
        cem_k=5,
        mppi_temp=10,
        vf_every=1600,
        vf_epoches=10,
        finetune_every=800,
        finetune_epoches=1,
    ):
        super(MPCAlgo, self).__init__(
            observation_shape=observation_shape,
            action_shape=action_shape,
            basis_dim=basis_dim,
            learning_rate=learning_rate,
            num_epoch=num_epoch,
            batch_size=batch_size,
            mpc_samples=mpc_samples,
            mpc_horizon=mpc_horizon,
            buffer_size=buffer_size,
            gamma=gamma,
            reweight=reweight,
            ensemble=ensemble,
            disagreement_coef=disagreement_coef,
            normalize_basis=normalize_basis,
            successs_update_rate=successs_update_rate,
            planner=planner,
            cem_iters=cem_iters,
            cem_k=cem_k,
            mppi_temp=mppi_temp,
            vf_every=vf_every,
            vf_epoches=vf_epoches,
            finetune_every=finetune_every,
            finetune_epoches=finetune_epoches
        )

        self.psi_net = None
        self.psi_optimizer = None
        self.device = torch.device('cuda')
        self.is_pixel = len(observation_shape) == 3
        self._build_model()
        self.buffer = RampRolloutBuffer(buffer_size, observation_shape, action_shape, basis_dim)
        self.r_mean = 0.0
        self.r_std = 1.0
        self.online_lsq = OnlineLeastSquare()
        # self.fix_w = None
        # self.dynamic_weight = None
        self.exploration_steps = max(int(self.basis_dim * 1.2), 2048)
        self.vf_trained = False

    def _build_model(self):
        obs_dim = np.prod(self.observation_shape)
        action_dim = np.prod(self.action_shape)
        if self.is_pixel:
            self.psi_net = CnnBasisNet(obs_dim, action_dim, self.basis_dim,
                self.mpc_horizon, hidden_dim=min(4096, self.basis_dim * 4),
                ensemble=self.ensemble, channel=self.observation_shape[2]).to(self.device)
            self.vf = CnnVfNet(obs_dim, action_dim, self.mpc_horizon).to(self.device)
            self.vf_target = CnnVfNet(obs_dim, action_dim, self.mpc_horizon).to(self.device)
        else:
            self.psi_net = MlpBasisNet(obs_dim, action_dim, self.basis_dim,
                self.mpc_horizon, hidden_dim=min(4096, self.basis_dim * 4),
                ensemble=self.ensemble).to(self.device)
            self.vf = MlpVfNet(obs_dim, action_dim, self.mpc_horizon).to(self.device)
            self.vf_target = MlpVfNet(obs_dim, action_dim, self.mpc_horizon).to(self.device)
        self.vf_target.load_state_dict(self.vf.state_dict())
        self.psi_optimizer = torch.optim.Adam(self.psi_net.parameters(), lr=self.learning_rate)
        self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), lr=self.learning_rate)


    def train_offline(self, rollout_dir, val_ratio=0.0):
        dataset = self.load_dataset(rollout_dir)
        if val_ratio > 0:
            train_len = int((1 - val_ratio) * len(dataset))
            test_len = len(dataset) - train_len
            train_set, test_set = random_split(dataset, (train_len, test_len))
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
            test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True, num_workers=4)
        else:
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        r_mean = torch.from_numpy(self.r_mean).to(self.device)
        r_std = torch.from_numpy(self.r_std).to(self.device)

        wandb.define_metric("offline/step")
        wandb.define_metric("offline/*", step_metric="offline/step")
        step = 0
        for _ in range(self.num_epoch):
            for data in tqdm(train_loader, total=len(train_loader), desc='Train MPC psi',
                             disable=os.environ.get("DISABLE_TQDM", False)):
                step += 1
                obs, action, reward, done = [d.to(self.device) for d in data]
                reward = (reward - r_mean[None, None]) / r_std[None, None]
                if self.is_pixel:
                    obs = obs / 255.0 * 2.0 - 1.0
                psi_hat = self.psi_net(obs[:, 0], action)
                decay = torch.from_numpy(
                    np.geomspace(1, self.gamma**(self.mpc_horizon-1), num=self.mpc_horizon)
                ).float().to(self.device)
                psi = torch.sum(reward * decay[None, :, None], 1)

                loss = torch.mean((psi_hat - psi[:, None]) ** 2, (0, 2)).sum()

                self.psi_optimizer.zero_grad()
                loss.backward()
                self.psi_optimizer.step()

                if step % 200 == 0:
                    wandb.log({
                        'offline/mpc_psi_train_loss': loss.item() / self.ensemble,
                        'offline/step': step,
                    })

                    if val_ratio > 0:
                        val_losses = []
                        for data in test_loader:
                            obs, action, reward, done = [d.to(self.device) for d in data]
                            reward = (reward - r_mean[None, None]) / r_std[None, None]
                            if self.is_pixel:
                                obs = obs / 255.0 * 2.0 - 1.0
                            with torch.no_grad():
                                psi_hat = self.psi_net(obs[:, 0], action)
                            decay = torch.from_numpy(
                                np.geomspace(1, self.gamma**(self.mpc_horizon-1), num=self.mpc_horizon)
                            ).float().to(self.device)
                            psi = torch.sum(reward * decay[None, :, None], 1)
                            loss = torch.mean((psi_hat - psi[:, None]) ** 2, (0, 2)).sum()
                            val_losses.append(loss.item())
                        wandb.log({'offline/mpc_psi_val_loss': np.mean(val_losses) / self.ensemble})

    def load_dataset(self, path):
        if os.path.isdir(path):
            datasets = []
            for file_path in os.listdir(path):
                if '.rollout' in file_path:
                    datasets.append(self.load_dataset(os.path.join(path, file_path)))
            wandb.run.summary['dataset_success_rate'] = np.mean([d.success_rate for d in datasets])
            concat_dataset = torch.utils.data.ConcatDataset(datasets)

            dataset_rewards = torch.cat([d.rollout['reward'] for d in datasets], 0)
            self.r_mean = torch.mean(dataset_rewards, dim=(0, 1)).float().numpy()
            self.r_std = torch.std(dataset_rewards, dim=(0, 1)).float().numpy()

            if not self.normalize_basis:
                self.r_mean = np.zeros_like(self.r_mean)
                self.r_std = np.ones_like(self.r_std)

            return concat_dataset
        else:
            rollout = torch.load(path)
            rollout = {k: torch.from_numpy(v) for k, v in rollout.items()}
            return TrajSliceDataset(rollout, self.mpc_horizon)

    def _add_to_buffer(self, obs, action, reward, done, phi):
        batch_size = 1
        assert len(obs) == batch_size
        self.buffer.add(obs[0], action[0], reward[0], done[0], phi[0])

    def update_lsq(self, reward, phi):
        batch_size = 1
        if not self.online_lsq.initialized:
            A, b = self.buffer.regression_data()
            scale = 1 / np.sqrt(self.reweight + b ** 2) if self.reweight > 0 else np.ones_like(b)
            self.online_lsq.initialize(A * scale[:, None], b * scale)
        else:
            scale = 1 / np.sqrt(self.reweight + reward ** 2) if self.reweight > 0 else np.ones_like(reward)
            phi = np.concatenate([phi, np.ones((batch_size, 1))], -1)
            self.online_lsq.update(phi * scale[:, None], reward * scale)


    def train_online(self, env, policy=None, projection_net=None, num_episodes=50, callbacks=[]):
        wandb.define_metric("online/step")
        wandb.define_metric("online/*", step_metric="online/step")

        self.buffer.reset()
        self.vf_trained = False

        num_envs = env.num_envs

        if projection_net and isinstance(projection_net, nn.Module):
            projection_net = projection_net.to(self.device)

        return_stat = np.zeros((num_episodes, num_envs))
        return_hat_stat = np.zeros((num_episodes, num_envs))
        value_stat = np.zeros((num_episodes, num_envs))
        value_hat_stat = np.zeros((num_episodes, num_envs))
        q_error_discount_stat = np.zeros((num_episodes, num_envs))
        disagreement_stat = np.zeros((num_episodes, num_envs))
        success_stat = np.zeros((num_episodes, num_envs))



        env.reset()
        total_steps = 0
        for episode_idx in trange(num_episodes, desc='Evaluation Episode',
                                  disable=os.environ.get("DISABLE_TQDM", False)):
            done = False
            step_idx = 0
            hist = dict(obs=[], action=[], reward=[])
            obs = env.reset()
            while not done:
                # draw action candidates from policy
                if obs.dtype == np.uint8:
                    obs = obs.astype(np.float32) / 255.0 * 2 - 1.0

                if self.buffer.total >= self.exploration_steps or self.online_lsq.initialized:
                    if policy:
                        action_samples = policy.predict(obs)[0].reshape(num_envs, 1, 1, -1)
                    else:
                        action_samples = np.stack([env.action_space.sample() for _ in
                                                   range(num_envs * self.mpc_samples * self.mpc_horizon)], 0)
                        action_samples = action_samples.reshape(num_envs, self.mpc_samples, self.mpc_horizon, -1)

                    # calculate optimal action, reward basis and w
                    action, w = self._predict(obs, action_samples)
                else:
                    action = np.stack([env.action_space.sample() for _ in range(num_envs)])
                    w = self.calc_w()

                if projection_net:
                    reward_basis = projection_net(
                        torch.from_numpy(obs).to(self.device),
                        torch.from_numpy(action).to(self.device),
                    ).detach().cpu().numpy()
                    reward_basis = (reward_basis - self.r_mean[None]) / self.r_std[None]
                else:
                    raise NotImplementedError
                    # reward_basis = reward_basis_hat
                old_obs = obs
                obs, reward, done, info = env.step(action)
                # self.set_dynamic_reweight(np.exp(-reward * self.reweight))
                # reward = np.array([item['original_reward'] for item in info])

                success = sum([item['success'] for item in info]) > 0
                if self.vf_trained:
                    self.online_lsq.adjust_update_rate(self.successs_update_rate)

                # update buffer to update w
                self._add_to_buffer(old_obs, action, reward, done, reward_basis)
                if self.buffer.total >= self.exploration_steps or self.online_lsq.initialized:
                    self.update_lsq(reward, reward_basis)


                done = np.any(done)

                # add experience to history for q stats
                hist['obs'].append(obs)
                hist['action'].append(action)


                hist['reward'].append(reward)

                # collect stats for metrics
                return_stat[episode_idx] += np.array(reward)
                return_hat_stat[episode_idx] += np.sum(reward_basis * w[None, :-1], 1) + w[None, -1]
                value_stat[episode_idx] += np.array(reward) * (self.gamma ** step_idx)
                value_hat_stat[episode_idx] += (np.sum(reward_basis * w[None, :-1], 1) + w[None, -1]) \
                                               * (self.gamma ** step_idx)

                step_idx += 1
                total_steps += 1

            if self.online_lsq.initialized and total_steps >= self.exploration_steps + self.vf_every:
                if self.finetune_every > 0 and total_steps % self.finetune_every:
                    fine_tune_loss = self.finetune_psi_networks()
                    wandb.log({'online/mpc_psi_finetune_loss': np.mean(fine_tune_loss),
                               "online/step": total_steps * num_envs})
                if self.vf_every > 0 and total_steps % self.vf_every:
                    on_policy_episodes = self.vf_every // self.buffer.episode_len
                    vf_err = np.mean(self.train_vf(on_policy_episodes))
                    wandb.log({"online/vf_err": vf_err, "online/step": total_steps * num_envs})
                    self.vf_trained = True

            # collect stats for metrics
            success_stat[episode_idx] = np.array([item['success'] for item in info])
            q_error_discount_stat[episode_idx], disagreement_stat[episode_idx] = \
                self.calc_q_error(hist['obs'], hist['action'], hist['reward'])

            wandb.log({
                "online/return": return_stat[episode_idx].mean(),
                "online/return_hat": return_hat_stat[episode_idx].mean(),
                "online/return_error": np.abs(return_stat[episode_idx] - return_hat_stat[episode_idx]).mean(),
                # "benchmark/value_error": np.abs(value_stat[episode_idx] - value_hat_stat[episode_idx]).mean(),
                "online/success_rate": success_stat[episode_idx].mean(),
                "online/q_error_discount": q_error_discount_stat[episode_idx].mean(),
                "online/disagreement_stat": disagreement_stat[episode_idx].mean(),
                "online/step": total_steps * num_envs
            })

            for callback in callbacks:
                callback(env, self, projection_net, self.calc_w())


        wandb.run.summary["return_mean"] = return_stat.mean()
        wandb.run.summary["return_max"] = return_stat.max()
        wandb.run.summary["return_std"] = return_stat.std()
        wandb.run.summary["success_rate"] = success_stat.mean()

    def _predict(self, obs, action_samples):
        batch_size, mpc_samples, mpc_horizon, action_dim = action_samples.shape
        obs = torch.from_numpy(obs).to(self.device)
        obs = obs.repeat(mpc_samples, *[1 for _ in range(len(self.observation_shape))])
        action_samples = torch.from_numpy(action_samples).to(self.device)
        decay = np.geomspace(1, self.gamma**(self.mpc_horizon-1), num=self.mpc_horizon, dtype=np.float32)
        w = self.calc_w()
        if self.planner == "random_shooting":
            action_samples = action_samples.reshape(batch_size * mpc_samples, mpc_horizon, action_dim)
            with torch.no_grad():
                psi_hat = self.psi_net(obs, action_samples)  # (batch_size * mpc_samples, ensemble, basis_dim)
                value = self.vf(obs, action_samples)  # (batch_size * mpc_samples, 1)
            psi_hat = psi_hat.reshape(batch_size, mpc_samples, self.ensemble, self.basis_dim).detach().cpu().numpy()
            value = value.reshape(batch_size, mpc_samples).detach().cpu().numpy()
            q_hat = np.sum(psi_hat * w[None, None, None, :-1], -1) + \
                w[None, None, None, -1] * decay.sum() # (batch_size, mpc_samples, self.ensemble)
            disagreement = q_hat.std(2) ** 2
            q_hat = q_hat.mean(2)
            if self.vf_trained:
                q_hat +=  self.gamma ** self.mpc_horizon * value
            if self.disagreement_coef > 0:
                q_hat -= disagreement * self.disagreement_coef
            best_indices = np.argmax(q_hat, axis=1) + np.arange(batch_size) * mpc_samples
            best_action = action_samples[best_indices, 0].detach().cpu().numpy()
        elif self.planner == "cem" or self.planner == "mppi":
            w_t = torch.from_numpy(w).to(self.device)
            decay_t = torch.from_numpy(decay).float().to(self.device)
            for _ in range(self.cem_iters):
                # Evaluate action sequences
                flat_action_samples = action_samples.reshape(batch_size * mpc_samples, mpc_horizon, action_dim)
                with torch.no_grad():
                    psi_hat = self.psi_net(obs, flat_action_samples)  # (batch_size * mpc_samples, ensemble, basis_dim)
                    value = self.vf(obs, flat_action_samples)  # (batch_size * mpc_samples, 1)
                psi_hat = psi_hat.reshape(batch_size, mpc_samples, self.ensemble, self.basis_dim)
                value = value.reshape(batch_size, mpc_samples)
                q_hat = torch.sum(psi_hat * w_t[None, None, None, :-1], -1) + w_t[None, None, None, -1] * decay_t.sum()
                disagreement = q_hat.std(2) ** 2
                q_hat = q_hat.mean(2)  # (batch_size, mpc_samples)
                if self.vf_trained:
                    q_hat += self.gamma ** self.mpc_horizon * value
                if self.disagreement_coef > 0:
                    q_hat -= disagreement * self.disagreement_coef
                # Fit new distribution
                if self.planner == "cem":
                    # Select top k actions
                    top_qs, top_inds = torch.topk(q_hat, self.cem_k, dim=1)
                    # (batch_size, k, mpc_horizon, action_dim)
                    top_action_samples = action_samples[torch.arange(batch_size)[:, None], top_inds] 
                    action_std, action_mean = torch.std_mean(top_action_samples, 1)
                elif self.planner == "mppi":
                    weights = F.log_softmax(q_hat * self.mppi_temp, dim=1).exp().float()[:, :, None, None]
                    action_mean = (weights * action_samples).sum(1)
                    action_var = (weights * (action_samples - action_mean[:, None, :, :]) ** 2).sum(1)
                    action_std = (action_var + 1e-8).sqrt()
                # Sample new action sequences
                dist = Normal(action_mean, action_std)
                action_samples = dist.sample((mpc_samples,)).transpose(0, 1)
            best_action = action_mean[:, 0].detach().cpu().numpy()
        return best_action, w

    def predict(self, obs):
        return self._predict(obs)[0]

    def calc_w(self):
        if not self.online_lsq.initialized:
            return np.ones(self.basis_dim + 1) / self.basis_dim
        # elif self.buffer_full:
        #     A = self.buffer[0]
        #     b = self.buffer[1]
        # else:
        #     A = self.buffer[0][:self.buffer_top]
        #     b = self.buffer[1][:self.buffer_top]
        # loss_weight = 1 / np.sqrt(self.reweight + b**2)
        # w = np.linalg.lstsq(A * loss_weight[:, None], b * loss_weight, rcond=None)[0]
        w = self.online_lsq.calculate()
        return w

    def calc_w_offline(self):
        # Offline least squares from buffer
        X, y = self.buffer.regression_data()
        reg = LinearRegression(fit_intercept=False).fit(X, y)
        return reg.coef_

    def calc_q_error(self, obs_hist, action_hist, reward_hist):
        # self.set_dynamic_reweight(0.4)

        obs_hist = np.array(obs_hist)
        action_hist = np.array(action_hist)
        reward_hist = np.array(reward_hist)
        num_envs = obs_hist.shape[1]
        _, _, action_dim = action_hist.shape

        batch = dict(obs=[], action=[], reward=[])
        decay = torch.from_numpy(
            np.geomspace(1, self.gamma**(self.mpc_horizon-1), num=self.mpc_horizon)
        ).float().to(self.device)
        w = torch.from_numpy(self.calc_w()).to(self.device)
        for i in range(0, len(obs_hist) - self.mpc_horizon):
            batch['obs'].append(obs_hist[i])  # t, num_env, dim
            batch['action'].append(action_hist[i: i + self.mpc_horizon])  # t, horizon, num_env, action_dim
            batch['reward'].append(reward_hist[i: i + self.mpc_horizon])  # t, horizon, num_env, 1

        batch = {k: torch.from_numpy(np.stack(v)).to(self.device) for k, v in batch.items()}

        with torch.no_grad():
            if self.is_pixel:
                batch['obs'] = batch['obs'] / 255.0 * 2 - 1.0
                obs = batch['obs'].permute(1, 0, 2, 3, 4).contiguous().view(-1, *self.observation_shape)
            else:
                obs = batch['obs'].permute(1, 0, 2).contiguous().view(-1, *self.observation_shape)
            action = batch['action'].permute(2, 0, 1, 3).contiguous().view(-1, self.mpc_horizon, action_dim)
            reward = batch['reward'].permute(2, 0, 1).contiguous().view(-1, self.mpc_horizon)
            psi_hat = self.psi_net(obs, action)  # num_env * t, ensemble, basis_dim
            q_hat = torch.sum(psi_hat * w[None, None, :-1], 2) + w[None, None, -1] * decay.sum()
            disagreement, q_hat = torch.var_mean(q_hat, 1, False)
            q_discount = torch.sum(reward * decay[None], 1)

            q_error_discount = torch.abs(q_hat - q_discount).view(num_envs, -1).mean(1).detach().cpu().numpy()
            disagreement = disagreement.view(num_envs, -1).mean(1).detach().cpu().numpy()

        return q_error_discount, disagreement

    def train_vf(self, on_policy_episodes):
        dataset = self.buffer.get_dataset(self.mpc_horizon + 1, on_policy_episodes)
        criteria = nn.MSELoss()
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
        loss_stat = []
        for _ in range(self.vf_epoches):
            for data in train_loader:
                obs, action, reward = [d.cuda() for d in data]
                with torch.no_grad():
                    vf_target = self.gamma * self.vf(obs[:, 1], action[:, 1:]) + reward[:, -1:]
                    vf_target.detach()
                vf_hat = self.vf(obs[:, 0], action[:, :-1])
                loss = criteria(vf_hat, vf_target)
                self.vf_optimizer.zero_grad()
                loss.backward()
                self.vf_optimizer.step()
                polyak_update(self.vf.parameters(), self.vf_target.parameters(), 0.005)
                loss_stat.append(loss.item())
        return np.array(loss_stat)

    def finetune_psi_networks(self):
        for g in self.psi_optimizer.param_groups:
            g['lr'] = self.learning_rate * 0.1

        dataset = self.buffer.get_dataset(self.mpc_horizon)
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

        loss_stat = []
        for _ in range(self.finetune_epoches):
            for data in train_loader:
                obs, action, reward = [d.cuda() for d in data]

                w_t = torch.from_numpy(self.calc_w()).to(self.device)
                decay = torch.from_numpy(
                    np.geomspace(1, self.gamma ** (self.mpc_horizon - 1), num=self.mpc_horizon)
                ).float().to(self.device)

                psi_hat = self.psi_net(obs[:, 0], action)
                q_hat = torch.sum(psi_hat * w_t[None, None, :-1], 2) + w_t[None, None, -1] * decay.sum()
                q_discount = torch.sum(reward[..., None] * decay[None, :, None], 1)
                loss = ((q_hat - q_discount) ** 2).sum(1).mean()

                self.psi_optimizer.zero_grad()
                loss.backward()
                self.psi_optimizer.step()

                loss_stat.append(loss.item() / self.ensemble)

        return np.array(loss_stat)

    @property
    def _state_attributes(self):
        return ['psi_net', 'psi_optimizer', 'vf', 'r_mean', 'r_std']

    # def set_dynamic_reweight(self, value):
    #     self.dynamic_weight = value
