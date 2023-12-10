from multiprocessing import Process
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
import gym
import pathlib
import os
import argparse
import re

from environments import make_env


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='ReachEnv-v2')
    parser.add_argument('--total_steps', type=int, default=800000)
    parser.add_argument('--n_envs', type=int, default=50)
    parser.add_argument('--threads', type=int, default=5)
    return parser.parse_args()


def train_policy(rank, env_id, total_steps):
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus[rank % len(gpus)]
    ckpt_dir = os.path.join('ckpts', env_id, f'policy_{rank}')
    data_dir = os.path.join('data', env_id, f'policy_{rank}')
    env = make_env(args.env_id, rank, 'train')
    env = Monitor(env, filename=data_dir, info_keywords=('success', ))

    is_pixel = 'Pixel' in env_id
    policy_type = "CnnPolicy" if is_pixel else "MlpPolicy"
    model = SAC(
        policy_type,
        env,
        learning_starts=500,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
    )
    model.learn(total_timesteps=total_steps, log_interval=32)
    model.save(ckpt_dir)


if __name__ == '__main__':
    args = get_args()
    pathlib.Path(os.path.join('ckpts', args.env_id)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join('data', args.env_id)).mkdir(parents=True, exist_ok=True)

    chunks = [range(i * args.threads, min(args.n_envs, args.threads * (i + 1)))
              for i in range((args.n_envs - 1) // args.threads + 1)]

    for chunk in chunks:
        processes = [Process(target=train_policy, args=(rank, args.env_id, args.total_steps)) for rank in chunk]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
