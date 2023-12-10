import gym
import numpy as np
from gym import ObservationWrapper

class CastObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if self.env.observation_space.dtype != np.uint8:
            self._observation_space = gym.spaces.Box(
                low=self.env.observation_space.low,
                high=self.env.observation_space.high, 
                shape=self.env.observation_space.shape,
                dtype=np.float32,
            )

    def observation(self, observation):
        if observation.dtype != np.uint8:
            return observation.astype(np.float32)
        else:
            return observation

    @property
    def goal(self):
        return self.unwrapped._get_goal()
