import numpy as np
import gym


class MazePixelEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MazePixelEnvWrapper, self).__init__(env)
        self.resolution = 84
        self._observation_space = gym.spaces.Box(
            high=255, low=0,
            shape=(self.resolution, self.resolution, 3),
            dtype=np.uint8,
        )
        self.state_obs = None

    def reset(self, **kwargs):
        self.state_obs = self.env.reset(**kwargs)
        return self.render('rgb_array')

    def step(self, action):
        self.state_obs, reward, done, info = self.env.step(action)
        obs = self.render('rgb_array')
        info['state_obs'] = self.state_obs
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.env.render(
                mode='rgb_array',
                width=self.resolution,
            )
        else:
            return self.env.render(mode)

    @property
    def goal(self):
        return self.env._get_target()
