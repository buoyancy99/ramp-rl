import gym
import numpy as np
from scipy.stats import multivariate_normal


class Point2DEnv(gym.Env):
    def __init__(self, seed=0):
        super(Point2DEnv, self).__init__()
        self.np_random, _ = self.seed(seed)
        self.init_pos = np.array([0., 0.])
        self.goal = (self.np_random.rand(2) - 0.5) * 2.0
        self.max_vel = 0.06
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, ))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2, ))
        self.perturbations = []

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return self.np_random, seed

    def reset(self):
        self.pos = (self.np_random.rand(2) - 0.5) * 2.0
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self.pos).astype(np.float32)

    def set_goal(self, goal):
        goal = np.array(goal, dtype=np.float32)
        assert len(goal) == 2 and np.all(goal**2 <= 1), "Invalid goal"
        self.goal = goal

    def add_perturbation(self, loc, var, coef):
        loc = np.array(loc, dtype=np.float32)
        var = np.array(var, dtype=np.float32)
        assert len(loc) == 2 and np.all(loc**2 <= 1), "Invalid perturbation pos"
        assert var >= 0, "Invalid perturbation variance"
        dist = multivariate_normal(loc, np.eye(2) * var)
        self.perturbations.append((dist, coef))

    @classmethod
    def get_reward(cls, obs, action, goal, perturbations=None):
        # distance to goal - action penalty
        reward = 1 - np.linalg.norm(obs - goal, axis=-1) - 0.01 * np.linalg.norm(action, axis=-1)
        if perturbations is not None:
            reward += sum([coef * dist.pdf(obs) for dist, coef in perturbations])
        # perturbation = sum([(1 - np.linalg.norm(obs - dist.mean, axis=-1)) * coef for dist, coef in perturbations])
        return reward

    def step(self, action):
        self.pos += self.max_vel * action
        obs = self._get_obs()
        reward = self.get_reward(obs, action, self.goal, self.perturbations)
        success = np.linalg.norm(self.pos - self.goal) < 0.05
        return obs, reward, False, {'success': success, 'is_success': success}
