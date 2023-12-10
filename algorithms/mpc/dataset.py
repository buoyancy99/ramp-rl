import torch


class TrajSliceDataset(torch.utils.data.Dataset):
    def __init__(self, rollout, horizon=6):
        super(TrajSliceDataset, self).__init__()
        self.rollout = rollout
        self.horizon = horizon
        self.num_episode, self.episode_len = rollout['obs'].shape[:2]
        self.success_rate = self.rollout['success'][:, -1, 0].float().mean().item()

    def __len__(self):
        return self.num_episode * (self.episode_len - self.horizon + 1)

    def __getitem__(self, idx):
        episode_idx = idx // (self.episode_len - self.horizon + 1)
        step_idx = idx % (self.episode_len - self.horizon + 1)
        obs = self.rollout['obs'][episode_idx, step_idx: step_idx + self.horizon].float()
        action = self.rollout['action'][episode_idx, step_idx: step_idx + self.horizon].float()
        reward = self.rollout['reward'][episode_idx, step_idx: step_idx + self.horizon].float()
        done = self.rollout['done'][episode_idx, step_idx: step_idx + self.horizon].float()
        return obs, action, reward, done


class BufferTrajSliceDataset(torch.utils.data.Dataset):
    def __init__(self, rollout, horizon=6):
        super(BufferTrajSliceDataset, self).__init__()
        self.rollout = {k: torch.from_numpy(v) for k, v in rollout.items()}
        self.horizon = horizon
        self.num_episode, self.episode_len = rollout['obs'].shape[:2]

    def __len__(self):
        return self.num_episode * (self.episode_len - self.horizon + 1)

    def __getitem__(self, idx):
        episode_idx = idx // (self.episode_len - self.horizon + 1)
        step_idx = idx % (self.episode_len - self.horizon + 1)
        obs = self.rollout['obs'][episode_idx, step_idx: step_idx + self.horizon].float()
        action = self.rollout['action'][episode_idx, step_idx: step_idx + self.horizon].float()
        reward = self.rollout['reward'][episode_idx, step_idx: step_idx + self.horizon].float()
        return obs, action, reward
