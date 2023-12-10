import os
import gym
import numpy as np
import pathlib
import argparse
import torch
import re
import wandb
import json
from tqdm import trange
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import SAC

from algorithms.mpc.mpc_algo import MPCAlgo
from environments import make_env
from environments.wrappers import RandomRewardWrapper, LearnedRewardWrapper


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs):
        return np.array([self.action_space.sample() for _ in range(len(obs))]), None


def run_mpc_experiment(config):
    rollout_dir = os.path.join("buffer", config['env_id'], f"{config['basis_type']}_{config['basis_dim']}")
    with open(os.path.join(rollout_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    algo = MPCAlgo(
        observation_shape=metadata['observation_shape'],
        action_shape=metadata['action_shape'],
        basis_dim=config['basis_dim'],
        num_epoch=config['num_offline_epoch'],
        mpc_samples=config['mpc_samples'],
        mpc_horizon=config['mpc_horizon'],
        gamma=config['gamma'],
        reweight=config['reweight'],
        ensemble=config['ensemble'],
        disagreement_coef=config['disagreement_coef'],
        normalize_basis=config['normalize'],
        planner=config['planner'],
        cem_iters=config['cem_iters'],
        cem_k=config['cem_k'],
        mppi_temp=config['mppi_temp'],
        vf_every=config['vf_every']
    )

    ckpt_path = os.path.join("ckpts", config['env_id'],
        f"esb_{config['ensemble']}_horizon{config['mpc_horizon']}" +
        f"_{config['basis_dim']}_gamma{config['gamma']}_mpc.pt")
    if pathlib.Path(ckpt_path).is_file():
        print('Loaded saved psi net ')
        ckpt = torch.load(ckpt_path)
        algo.psi_net.load_state_dict(ckpt['psi_net'])
        algo.r_mean = ckpt['r_mean']
        algo.r_std = ckpt['r_std']
    else:
        algo.train_offline(rollout_dir, val_ratio=config["validation_ratio"])
        ckpt = dict(psi_net=algo.psi_net.state_dict(), r_mean=algo.r_mean, r_std=algo.r_std)
        torch.save(ckpt, ckpt_path)

    env = DummyVecEnv([lambda: make_env(config['env_id'], 0, config['split'])])
    if config['basis_type'] == "rand":
        projection_net = RandomRewardWrapper(make_env(config['env_id'], config['seed'],
                                                      config['split']), config['basis_dim']).net
    elif config['basis_type'] == "learned":
        state_dict = torch.load(os.path.join("buffer", config['env_id'],
                                             f"{config['basis_type']}_{config['basis_dim']}", "feature.pt"))
        projection_net = LearnedRewardWrapper(
            make_env(config['env_id'], 0, config['split']),
            config['basis_dim'], state_dict=state_dict
        ).net

        # algo.fix_w = torch.load(os.path.join(rollout_dir, 'last_w.pt'))['weight'][config['seed].cpu().detach().numpy()

    if config['eval_policy'] == "online":
        print('==== Online evaluation using meta mpc ====')
        algo.train_online(env, projection_net=projection_net,
                          num_episodes=config['num_episodes'])

    if config['eval_policy'] == "random":
        print('==== Off policy evaluation using random policy ====')
        algo.train_online(
            env,
            policy=RandomPolicy(env.action_space),
            projection_net=projection_net,
            num_episodes=config['num_episodes']
        )

    if config['eval_policy'] == "expert":
        print('= Off policy evaluation using single goal policy ==')
        algo.train_online(
            env,
            policy=SAC.load(os.path.join("ckpts", config['env_id'], f"{config['env_id']}_{config['seed']}")),
            projection_net=projection_net,
            num_episodes=config['num_episodes']
        )
