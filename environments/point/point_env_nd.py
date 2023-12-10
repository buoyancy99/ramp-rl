import gym
import numpy as np
from scipy.stats import multivariate_normal


class PointNdEnv(gym.Env):
    def __init__(self, dim, seed=0):
        super(PointNdEnv, self).__init__()
        self.dim = dim
        self.np_random, _ = self.seed(seed)

        self.init_pos = np.zeros(self.dim, dtype=np.float32)
        self.goal = (self.np_random.rand(self.dim) - 0.5) * 2.0
        self.max_vel = 0.06
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.dim, ))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.dim, ))

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return self.np_random, seed

    def reset(self):
        self.pos = (self.np_random.rand(self.dim) - 0.5) * 2.0
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self.pos).astype(np.float32)

    def set_goal(self, goal):
        goal = np.array(goal, dtype=np.float32)
        assert len(goal) == self.dim and np.all(goal**2 <= 1), "Invalid goal"
        self.goal = goal

    def get_reward(self, obs, action):
        # distance to goal - action penalty
        reward = 1 - np.linalg.norm(obs - self.goal, axis=-1) - \
            0.01 * np.linalg.norm(action, axis=-1)
        return reward

    def step(self, action):
        self.pos += self.max_vel * action
        obs = self._get_obs()
        reward = self.get_reward(obs, action)
        success = np.linalg.norm(self.pos - self.goal) < 0.05
        return obs, reward, False, {'success': success, 'is_success': success}


class PointNdPerturbedEnv(PointNdEnv):
    def __init__(self, dim, seed=0, n_perturbations=0):
        super().__init__(dim=dim, seed=seed)
        self.perturbations = []
        for i in range(n_perturbations):
            self.add_random_perturbation()
    
    def add_random_perturbation(self):
        loc = (self.np_random.rand(self.dim) - 0.5) * 2 # [-1.0, 1.0]
        var = self.np_random.rand() * 0.1               # [ 0.0, 0.1]
        coef = (self.np_random.rand() - 0.5) * 0.6      # [-0.3, 0.3]
        self.add_perturbation(loc, var, coef)

    def add_perturbation(self, loc, var, coef):
        loc = np.array(loc, dtype=np.float32)
        var = np.array(var, dtype=np.float32)
        assert len(loc) == self.dim and np.all(loc**2 <= 1), "Invalid perturbation pos"
        assert var >= 0, "Invalid perturbation variance"
        dist = multivariate_normal(loc, np.eye(self.dim) * var)
        self.perturbations.append((dist, coef))
    
    def get_reward(self, obs, action):
        # distance to goal - action penalty
        reward = super().get_reward(obs, action)
        perturbation = sum([dist.pdf(obs) * coef for dist, coef in self.perturbations])
        return reward + perturbation

