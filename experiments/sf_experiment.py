import os
import numpy
import json
import pathlib

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
from tqdm import tqdm

from environments import make_env
from algorithms.sf import SfCqlAlgo
from algorithms.sf import SfAlgo
from environments.wrappers import RandomRewardWrapper, LearnedRewardWrapper


def run_sf_experiment(config):
    assert config['basis_type'] == 'learned'
    rollout_dir = os.path.join("buffer", config['env_id'], f"{config['basis_type']}_{config['basis_dim']}")
    rollout_names = [file_name for file_name in os.listdir(rollout_dir) if '.rollout' in file_name]
    ckpt_dir = os.path.join('ckpts', config['env_id'])
    policy_ckpt_paths = [str(x) for x in pathlib.Path(ckpt_dir).iterdir() if '.zip' in str(x) and 'test' not in str(x)]

    with open(os.path.join(rollout_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    with open(os.path.join("buffer", config['env_id'], "goals.json"), 'r') as f:
        goals = json.load(f)
    policy_goals = [goals[path.split('/')[-1].split('_')[-1][:-4] + '.rollout'] for path in policy_ckpt_paths]

    observation_shape = metadata['observation_shape']
    action_shape = metadata['action_shape']

    algo = SfAlgo(
        observation_shape,
        action_shape,
        arch='512-512',
        num_pi=len(policy_ckpt_paths),
        basis_dim=config['basis_dim'],
        reweight=config['reweight'],
    )
    qf1_state_dicts = []
    qf2_state_dicts = []

    ckpt_path = os.path.join("ckpts", config['env_id'], f"{config['env_id']}_sf.pt")

    if pathlib.Path(ckpt_path).is_file():
        algo = SfAlgo.load(ckpt_path)
    else:
        for policy_ckpt_path in tqdm(policy_ckpt_paths, desc='training q functions for sf',
                                     disable=os.environ.get("DISABLE_TQDM", False)):
            cql_algo = SfCqlAlgo(
                observation_shape=observation_shape,
                action_shape=action_shape,
                basis_dim=config["basis_dim"],
                goal_dim=0,
                arch=algo.arch,
            )
            policy = SAC.load(policy_ckpt_path).policy.actor

            cql_algo.train_offline(cql_algo.load_dataset(rollout_dir), policy)
            qf1_state_dicts.append(cql_algo.target_qf1.cpu().network.state_dict())
            qf2_state_dicts.append(cql_algo.target_qf2.cpu().network.state_dict())

        algo.qf1s.load_state_dict(algo.ensemble_state_dict(qf1_state_dicts))
        algo.qf2s.load_state_dict(algo.ensemble_state_dict(qf2_state_dicts))

        algo.save(ckpt_path)

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
    env = DummyVecEnv([lambda: make_env(config['env_id'], 0, config['split'])])
    algo.train_online(env, num_episodes=config['num_episodes'], projection_net=projection_net,
                      policy_goals=policy_goals, relabel_frac=config['relabel_frac'])

