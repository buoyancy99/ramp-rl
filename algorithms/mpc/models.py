import torch
import torch.nn as nn


class LstmBasisNet(nn.Module):
    def __init__(self, obs_dim, action_dim, basis_dim, hidden_dim=256, num_layers=2):
        super(LstmBasisNet, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.basis_dim = basis_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * num_layers),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(action_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, basis_dim)
        )

    def forward(self, obs, actions):
        """
        :param obs: Size(batch_size, *obs_shape)
        :param actions: Size(batch_size, seq_len, *action_shape)
        :return: rewards_hat: Size(batch_size, basis_dim)
        """
        output, _ = self.lstm(actions, self._init_hidden(obs))
        reward_hat = self.reward_head(output)
        reward_hat = torch.sum(reward_hat, 1)
        return reward_hat

    def _init_hidden(self, obs):
        batch_size = obs.shape[0]
        h_0 = self.obs_encoder(obs).view(batch_size, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c_0 = torch.zeros_like(h_0)
        return h_0, c_0


class MlpBasisNet(nn.Module):
    def __init__(self, obs_dim, action_dim, basis_dim, seq_len, hidden_dim=4096, ensemble=2):
        super(MlpBasisNet, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.basis_dim = basis_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.ensemble = ensemble
        input_dim = obs_dim + action_dim * seq_len
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim * ensemble, hidden_dim * ensemble // 2, 1, groups=ensemble),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * ensemble // 2, hidden_dim * ensemble, 1, groups=ensemble),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * ensemble, basis_dim * ensemble, 1, groups=ensemble)
        )

    def forward(self, obs, actions):
        """
        :param obs: Size(batch_size, *obs_shape)
        :param actions: Size(batch_size, seq_len, *action_shape)
        :return: psi_hat: Size(batch_size, ensemble, basis_dim)
        """
        batch_size = obs.shape[0]
        x = torch.cat([obs.flatten(1), actions.flatten(1)], 1)
        x = x.repeat(1, self.ensemble)[:, :, None]
        x = self.mlp(x)[:, :, 0]
        x = x.reshape(batch_size, self.ensemble, self.basis_dim)

        return x


class CnnBasisNet(nn.Module):
    def __init__(self, obs_dim, action_dim, basis_dim, seq_len, hidden_dim=4096, ensemble=2, channel=3):
        super(CnnBasisNet, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.basis_dim = basis_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.ensemble = ensemble

        self.cnn = nn.Sequential(
            nn.Conv2d(channel * ensemble, 32 * ensemble, kernel_size=8, stride=4, groups=ensemble),
            nn.ReLU(),
            nn.Conv2d(32 * ensemble, 64 * ensemble, kernel_size=4, stride=2, groups=ensemble),
            nn.ReLU(),
            nn.Conv2d(64 * ensemble, 64 * ensemble, kernel_size=3, stride=1, groups=ensemble)
        )

        self.action_mlp = nn.Conv1d(action_dim * seq_len * ensemble, 1024 * ensemble, 1, groups=ensemble)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d((64 * 7 * 7 + 1024) * ensemble, hidden_dim * ensemble // 2, 1, groups=ensemble),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * ensemble // 2, hidden_dim * ensemble, 1, groups=ensemble),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * ensemble, basis_dim * ensemble, 1, groups=ensemble),
        )

    def forward(self, obs, actions):
        """
        :param obs: Size(batch_size, *obs_shape)
        :param actions: Size(batch_size, seq_len, *action_shape)
        :return: psi_hat: Size(batch_size, ensemble, basis_dim)
        """
        batch_size = obs.shape[0]
        obs = obs.permute(0, 3, 1, 2)
        obs = obs.repeat(1, self.ensemble, 1, 1).contiguous()
        img_feature = self.cnn(obs).view(batch_size, self.ensemble, -1)
        actions = actions.flatten(1).repeat(1, self.ensemble)[:, :, None]
        action_feature = self.action_mlp(actions).view(batch_size, self.ensemble, -1)
        x = torch.cat([img_feature, action_feature], 2).reshape(batch_size, -1, 1)
        x = self.mlp(x).reshape(batch_size, self.ensemble, self.basis_dim)

        return x


class MlpVfNet(nn.Module):
    def __init__(self, obs_dim, action_dim, seq_len, hidden_dim=128):
        super(MlpVfNet, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        input_dim = obs_dim + action_dim * seq_len
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, actions):
        """
        :param obs: Size(batch_size, *obs_shape)
        :param actions: Size(batch_size, seq_len, *action_shape)
        :return: psi_hat: Size(batch_size, ensemble, basis_dim)
        """
        x = torch.cat([obs.flatten(1), actions.flatten(1)], 1)
        return self.mlp(x)


class CnnVfNet(nn.Module):
    def __init__(self, obs_dim, action_dim, seq_len, hidden_dim=128, channel=3):
        super(CnnVfNet, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        input_dim = obs_dim + action_dim * seq_len
        self.cnn = nn.Sequential(
            nn.Conv2d(channel, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
        )

        self.action_mlp = nn.Sequential(nn.Linear(action_dim * seq_len, hidden_dim), nn.ReLU())

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, actions):
        """
        :param obs: Size(batch_size, *obs_shape)
        :param actions: Size(batch_size, seq_len, *action_shape)
        :return: psi_hat: Size(batch_size, ensemble, basis_dim)
        """
        obs = self.cnn(obs.permute(0, 3, 1, 2))
        action = self.action_mlp(actions)
        x = torch.cat([obs.flatten(1), action.flatten(1)], 1)
        return self.mlp(x)
