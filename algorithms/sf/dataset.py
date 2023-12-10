import torch


class SfCqlReplayDataset(torch.utils.data.Dataset):
    def __init__(self, rollout):
        super(SfCqlReplayDataset, self).__init__()
        self.rollout = rollout
        self.num_episode, self.episode_len = rollout['obs'].shape[:2]
        self.rollout['action'] = torch.clip(self.rollout['action'], min=-0.98, max=0.98)
    def __len__(self):
        return self.num_episode * (self.episode_len - 1)

    def __getitem__(self, idx):
        episode_idx = idx // (self.episode_len - 1)
        step_idx = idx % (self.episode_len - 1)
        observation = self.rollout['obs'][episode_idx, step_idx].float()
        action = self.rollout['action'][episode_idx, step_idx].float()
        next_observation = self.rollout['obs'][episode_idx, step_idx + 1].float()
        done = self.rollout['done'][episode_idx, step_idx].float()
        reward = self.rollout['reward'][episode_idx, step_idx].float()
        goal = self.rollout['goal'][episode_idx, step_idx].float()
        return observation, action, next_observation, done, reward, goal
