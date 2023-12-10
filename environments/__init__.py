from gym.envs.registration import register
import gym

gym.logger.set_level(40)
import mujoco_py
import re
import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

from .wrappers import (
    CastObsWrapper,
    MultiviewMetaworldPixelEnvWrapper,
)

ml45_train = metaworld.ML45(seed=0)
ml45_test = metaworld.ML45(seed=9999)
env_episode_lens = {}

register(
    id="Point2dEnv-v1",
    entry_point="environments.point.point_env_nd:PointNdEnv",
    max_episode_steps=50,
    kwargs={"dim": 2},
)
env_episode_lens["Point2dEnv-v1"] = 50

register(
    id="Point2dPerturbedEnv-v1",
    entry_point="environments.point.point_env_nd:PointNdPerturbedEnv",
    max_episode_steps=50,
    kwargs={"dim": 2},
)
env_episode_lens["Point2dPerturbedEnv-v1"] = 50

register(
    id="Point3dEnv-v1",
    entry_point="environments.point.point_env_nd:PointNdEnv",
    max_episode_steps=50,
    kwargs={"dim": 3},
)
env_episode_lens["Point3dEnv-v1"] = 50

register(
    id="Point4dEnv-v1",
    entry_point="environments.point.point_env_nd:PointNdEnv",
    max_episode_steps=50,
    kwargs={"dim": 4},
)
env_episode_lens["Point4dEnv-v1"] = 50

for name in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys():
    cls_str = "".join([s.capitalize() for s in name[:-15].split("-")])
    env_episode_lens[f"{cls_str}Env-v2"] = 128
    env_episode_lens[f"{cls_str}PixelEnv-v2"] = 128

register(
    id="DclawEnv-v1",
    entry_point="environments.dclaw.dclaw_turn_env:DClawTurnEnv",
    max_episode_steps=200,
)
env_episode_lens["DclawEnv-v1"] = 200


def make_env(env_id, rank, split="train"):
    cls_name = "-".join([s.lower() for s in re.findall("[A-Z][^A-Z]*", env_id.replace("Pixel", "")[:-6])]) + "-v2"

    if cls_name in ml45_train.train_classes.keys():
        if split == "test":
            env = ml45_train.train_classes[cls_name]()
            task = [task for task in ml45_train.train_tasks if task.env_name == cls_name][rank]
        else:
            env = ml45_test.train_classes[cls_name]()
            task = [task for task in ml45_test.train_tasks if task.env_name == cls_name][rank]
        env.set_task(task)
        env._get_goal = env._get_pos_goal
        env = gym.wrappers.TimeLimit(env, max_episode_steps=env_episode_lens[env_id])
        if "Pixel" in env_id:
            # env = MetaworldPixelEnvWrapper(env, frame_stack=4)
            env = MultiviewMetaworldPixelEnvWrapper(env)

    else:
        if split == "test":
            rank += 9999
        else:
            env = gym.make(env_id, seed=rank)

    return CastObsWrapper(env)
