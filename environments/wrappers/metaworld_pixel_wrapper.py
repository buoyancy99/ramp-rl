import numpy as np
import gym

class MultiviewMetaworldPixelEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MultiviewMetaworldPixelEnvWrapper, self).__init__(env)
        # self.camera_names = ["corner", "corner2", "corner3"]
        self.camera_names = ["corner", "corner3"]
        self.resolution = 84
        self._observation_space = gym.spaces.Box(
            high=255, low=0,
            shape=(self.resolution, self.resolution, 3*len(self.camera_names)),
            dtype=np.uint8
        )
        self.state_obs = None
        
        # Render to make sure the first frame has the correct goal
        self.env.sim.render(
            camera_name='corner3',
            width=self.resolution,
            height=self.resolution,
            depth=False,
        )

    def reset(self, **kwargs):
        self.state_obs = self.env.reset(**kwargs)
        obs = self.render_multiview()
        return obs

    def step(self, action):
        self.state_obs, reward, done, info = self.env.step(action)
        obs = self.render_multiview()
        info['state_obs'] = self.state_obs
        return obs, reward, done, info
    
    def render_multiview(self):
        images = []
        for camera_name in self.camera_names:
            images.append(self.env.sim.render(
                camera_name=camera_name,
                width=self.resolution,
                height=self.resolution,
                depth=False,
            ))
        return np.concatenate(images, -1)


    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            image = self.env.sim.render(
                camera_name='corner3',
                width=self.resolution,
                height=self.resolution,
                depth=False,
            )
            return image
        else:
            return self.env.render(mode)

    @property
    def goal(self):
        return self.env._get_goal()

    def set_task(self, task):
        self.env.set_task(task)


class MetaworldPixelEnvWrapper(gym.Wrapper):
    def __init__(self, env, frame_stack=1):
        super(MetaworldPixelEnvWrapper, self).__init__(env)
        self.frame_stack = frame_stack
        self.frames = None
        self.resolution = 84
        self._observation_space = gym.spaces.Box(
            high=255, low=0,
            shape=(self.resolution, self.resolution, 3*self.frame_stack),
            dtype=np.uint8
        )
        self.state_obs = None
        
        # Render to make sure the first frame has the correct goal
        self.env.sim.render(
            camera_name='corner3',
            width=self.resolution,
            height=self.resolution,
            depth=False,
        )

    @property
    def stacked_obs(self):
        assert len(self.frames) == self.frame_stack
        return np.concatenate(self.frames, -1)

    def reset(self, **kwargs):
        self.state_obs = self.env.reset(**kwargs)
        obs = self.render('rgb_array')
        self.frames = [obs for _ in range(self.frame_stack)]
        return self.stacked_obs

    def step(self, action):
        self.state_obs, reward, done, info = self.env.step(action)
        obs = self.render('rgb_array')
        self.frames.pop(0)
        self.frames.append(obs)
        info['state_obs'] = self.state_obs
        return self.stacked_obs, reward, done, info

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            image = self.env.sim.render(
                camera_name='corner3',
                width=self.resolution,
                height=self.resolution,
                depth=False,
            )
            return image
        else:
            return self.env.render(mode)

    @property
    def goal(self):
        return self.env._get_goal()

    def set_task(self, task):
        self.env.set_task(task)
