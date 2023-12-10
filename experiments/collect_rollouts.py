from multiprocessing import Process
import numpy as np
from stable_baselines3 import SAC
from tqdm import trange, tqdm
from queue import Queue
import os
import re
import torch
import torch.nn as nn
import pathlib
import gym
import argparse
import json
from environments.wrappers import RandomRewardWrapper, PolynomialRewardWrapper, DummyRewardWrapper, LearnedRewardWrapper
from environments import make_env

success_stat_queue = Queue()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='ReachEnv-v2')
    parser.add_argument('--basis_type', type=str, choices=['rand', 'poly', 'learned'], default='rand')
    parser.add_argument('--steps', type=int, default=32000)
    parser.add_argument('--basis_dim', type=int, default=2048)
    parser.add_argument('--threads', type=int, default=5)
    parser.add_argument('--eps', type=float, default=0.75)
    return parser.parse_args()


def collect_rollout(rank, ckpt_path, rollout_dir, env_id, wrapper, total_steps, eps):
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus[rank % len(gpus)]

    policy = SAC.load(ckpt_path)
    use_pixel = 'Pixel' in env_id

    env = make_env(env_id, rank, 'train')  # fixme  maybe use rank later on
    env = wrapper(env)
    n_rewards = env.n_rewards

    obs_shape = env.observation_space.shape
    obs_dtype = env.observation_space.dtype

    obs = env.reset()
    goal_dim = len(env.goal.flatten())

    if not rank:
        with open(os.path.join(rollout_dir, 'metadata.json'), 'w') as f:
            metadata = dict(
                observation_shape=obs_shape,
                action_shape=env.action_space.shape,
                goal_dim=goal_dim
            )
            json.dump(metadata, f)

    rollout = dict(
        obs=np.zeros((total_steps,) + obs_shape, dtype=obs_dtype),
        action=np.zeros((total_steps,) + env.action_space.shape, dtype=env.action_space.dtype),
        reward=np.zeros((total_steps, n_rewards), dtype=np.float32),
        done=np.zeros((total_steps, 1), dtype=np.bool_),
        success=np.zeros((total_steps, 1), dtype=np.bool_),
        original_reward=np.zeros((total_steps, 1), dtype=np.float32),
        goal=np.zeros((total_steps, goal_dim), dtype=np.float32),
    )

    if use_pixel:
        rollout['state_obs'] = np.zeros(
            (total_steps, env.unwrapped.observation_space.shape[0]), 
            dtype=np.float32,
        )

    num_episodes = 0

    for i in tqdm(range(total_steps), disable=os.environ.get("DISABLE_TQDM", False)):
        state_obs = env.state_obs if use_pixel else obs
        action = policy.predict(obs)[0]
        if np.random.rand() < eps:
            action = env.action_space.sample()

        if use_pixel:
            rollout['state_obs'][i] = state_obs
        rollout['obs'][i] = obs
        rollout['action'][i] = action
        obs, reward, done, info = env.step(action)

        rollout['reward'][i] = reward
        rollout['done'][i] = done
        rollout['original_reward'][i] = info['original_reward']
        rollout['goal'][i] = env.goal

        if 'success' in info.keys():
            rollout['success'][i] = info['success']

        if done:
            obs = env.reset()
            num_episodes += 1

    rollout = {k: v.reshape(num_episodes, -1, *v.shape[1:]) for k, v in rollout.items()}
    rollout_path = os.path.join(rollout_dir, f'{rank}.rollout')
    torch.save(rollout, rollout_path)

    print('success_rate: ', rollout['success'].mean())

    success_stat_queue.put(rollout['success'].mean().item())


def learn_features(env_id, wrapper, rollout_dir):
    datasets = []
    is_pixel = 'Pixel' in env_id

    rand_rollout_dir = rollout_dir.split('/')
    rand_rollout_dir[-1] = rand_rollout_dir[-1].replace('learned', 'rand')
    rand_rollout_dir = '/'.join(rand_rollout_dir)

    rollout_names = [name for name in os.listdir(rand_rollout_dir) if '.rollout' in name]

    rollout_observations = []
    rollout_actions = []
    rollouts = []

    num_heads = len(rollout_names)
    for i, rollout_name in enumerate(rollout_names):
        rollout = torch.load(os.path.join(rand_rollout_dir, rollout_name))
        rollouts.append(rollout)
        num_episodes, episode_len = rollout['obs'].shape[:2]
        obs_shape = rollout['obs'].shape[2:]
        _, _, action_dim = rollout['action'].shape
        observations = torch.from_numpy(rollout['obs']).reshape(num_episodes * episode_len, *obs_shape).float()
        actions = torch.from_numpy(rollout['action']).reshape(num_episodes * episode_len, action_dim).float()
        rollout_observations.append(observations)
        rollout_actions.append(actions)
        rewards = torch.from_numpy(rollout['original_reward']).reshape(-1, rollout['original_reward'].shape[-1])
        reward_indices = torch.full([len(rewards), 1], i, dtype=torch.long)
        datasets.append(torch.utils.data.TensorDataset(observations, actions, rewards, reward_indices))

    datasets = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=256, shuffle=True, num_workers=4)

    env = make_env(env_id, 0, 'train')  # fixme  maybe use rank later on
    env = wrapper(env)
    n_rewards = env.n_rewards
    model = env.net.cuda()
    last_layer = nn.Linear(env.n_rewards, num_heads, bias=False).cuda()

    optimizer = torch.optim.Adam(list(model.parameters()) + list(last_layer.parameters()))

    for _ in trange(4, desc='Learning features'):
        for i, data in enumerate(dataloader):
            obs, action, reward, reward_idx = [v.cuda() for v in data]
            if is_pixel:
                obs = obs.float() / 255.0 * 2 - 1.0
            reward_basis = model(obs, action)
            reward_hat = last_layer(reward_basis)
            reward_hat = torch.gather(reward_hat, 1, reward_idx)
            loss = torch.nn.functional.mse_loss(reward_hat, reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for i, (rollout_name, rollout, observations, actions) in \
            enumerate(zip(rollout_names, rollouts, rollout_observations, rollout_actions)):
        rollout['reward'] = np.zeros((np.prod(rollout['reward'].shape[:2]), n_rewards), dtype=np.float32)
        original_rewards = torch.from_numpy(rollout['original_reward']).reshape(-1, 1).cuda()
        sample_id = 0
        error_stat = []
        for obs, action, original_reward in \
                zip(observations.split(256), actions.split(256), original_rewards.split(256)):
            batch_size = len(obs)
            if is_pixel:
                obs = obs.float() / 255.0 * 2 - 1.0
            obs = obs.cuda()
            action = action.cuda()
            with torch.no_grad():
                reward_basis = model(obs, action)
                reward_hat = last_layer(reward_basis)[:, i]
                error = torch.abs(reward_hat - original_reward[:, 0]).detach().cpu().numpy()
                error_stat.extend(error)
                rollout['reward'][sample_id:sample_id + batch_size] = reward_basis.detach().cpu().numpy()
            sample_id += batch_size
        assert sample_id == len(rollout['reward'])
        error_stat = np.array(error_stat)
        print(f'feature learning {i}.rollout, error mean {error_stat.mean()}, '
              f'std {error_stat.std()}, max {error_stat.max()}')
        rollout['reward'] = rollout['reward'].reshape(num_episodes, episode_len, n_rewards)
        torch.save(rollout, os.path.join(rollout_dir, rollout_name))

    torch.save(model.state_dict(), os.path.join(rollout_dir, 'feature.pt'))
    torch.save(last_layer.state_dict(), os.path.join(rollout_dir, 'last_w.pt'))

    # # eval
    # env = LearnedRewardWrapper(make_env(env_id, 0, 'train'), env.n_rewards, state_dict=model.state_dict())
    # obs = env.reset()
    # for _ in range(256):
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    #     reward_hat = last_layer(torch.from_numpy(reward)[None].cuda())
    #     original_reward = info['original_reward']
    #     print(reward_hat - original_reward)


if __name__ == '__main__':
    args = get_args()
    rollout_dir = os.path.join('buffer', args.env_id, f'{args.basis_type}_{args.eps}_{args.basis_dim}')
    rand_rollout_dir = os.path.join('buffer', args.env_id, f'rand_{args.eps}_{args.basis_dim}')
    pathlib.Path(rollout_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(rand_rollout_dir).mkdir(parents=True, exist_ok=True)

    wrapper_cls = dict(
        rand=lambda e: RandomRewardWrapper(e, args.basis_dim, device='cpu'),
        learned=lambda e: LearnedRewardWrapper(e, args.basis_dim, device='cpu'),
        poly=PolynomialRewardWrapper,
        none=DummyRewardWrapper,
    )

    wrapper = wrapper_cls[args.basis_type]

    if args.basis_type == 'rand' or \
        args.basis_type == 'poly' or \
        (args.basis_type == 'learned' and not len(os.listdir(rand_rollout_dir))):
        ckpt_dir = os.path.join('ckpts', args.env_id)
        ckpt_paths = [str(x) for x in pathlib.Path(ckpt_dir).iterdir() if '.zip' in str(x) and 'test' not in str(x)]
        ckpt_paths = sorted(ckpt_paths, key=lambda x : int(x.split("_")[-1][:-4]))

        num_ckpts = len(ckpt_paths)
        chunks = [range(i * args.threads, min(num_ckpts, args.threads * (i + 1)))
                  for i in range((num_ckpts - 1) // args.threads + 1)]

        for chunk in chunks:
            processes = [Process(target=collect_rollout,
                                 args=(i, ckpt_paths[i], rand_rollout_dir, args.env_id, wrapper, args.steps, args.eps))
                         for i in chunk]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

        with open(os.path.join(rand_rollout_dir, 'success_rate.json'), 'w') as f:
            json.dump(list(success_stat_queue.queue), f)

    if args.basis_type == 'learned':
        print('learning features')
        with open(os.path.join(rand_rollout_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        with open(os.path.join(rollout_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        learn_features(args.env_id, wrapper, rollout_dir)

    rollout_names = [name for name in os.listdir(rollout_dir) if '.rollout' in name]
    goals = {}
    for rollout_name in tqdm(rollout_names, desc='collecting goal info', disable=os.environ.get("DISABLE_TQDM", False)):
        rollout = torch.load(os.path.join(rollout_dir, rollout_name))
        goal_dim = rollout['goal'].shape[-1]
        rollout['goal'] = rollout['goal'].reshape(-1, goal_dim)
        goal = rollout['goal'][0]
        assert np.array_equal(rollout['goal'], goal[None] * np.ones_like(rollout['goal']))
        goals[rollout_name] = goal.tolist()

    stat_path = os.path.join("buffer", args.env_id, "goals.json")
    with open(stat_path, 'w') as f:
        json.dump(goals, f)


