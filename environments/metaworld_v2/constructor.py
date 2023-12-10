import numpy as np
import gym
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN

# from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
#
# env_names = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.keys()
# for name in env_names:
#     task = name[:-15]
#     cls_str = ''.join([s.capitalize() for s in task.split('-')])
#     print(f"{cls_str}Env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['{name}']")

# for k, v in ALL_V2_ENVIRONMENTS_GOAL_HIDDEN.items():
#     v._get_goal = v._get_pos_goal
#
#
# AssemblyEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['assembly-v2-goal-hidden']
# BasketballEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['basketball-v2-goal-hidden']
# BinPickingEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['bin-picking-v2-goal-hidden']
# BoxCloseEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['box-close-v2-goal-hidden']
# ButtonPressTopdownEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['button-press-topdown-v2-goal-hidden']
# ButtonPressTopdownWallEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['button-press-topdown-wall-v2-goal-hidden']
# ButtonPressEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['button-press-v2-goal-hidden']
# ButtonPressWallEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['button-press-wall-v2-goal-hidden']
# CoffeeButtonEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['coffee-button-v2-goal-hidden']
# CoffeePullEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['coffee-pull-v2-goal-hidden']
# CoffeePushEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['coffee-push-v2-goal-hidden']
# DialTurnEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['dial-turn-v2-goal-hidden']
# DisassembleEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['disassemble-v2-goal-hidden']
# DoorCloseEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['door-close-v2-goal-hidden']
# DoorLockEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['door-lock-v2-goal-hidden']
# DoorOpenEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['door-open-v2-goal-hidden']
# DoorUnlockEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['door-unlock-v2-goal-hidden']
# HandInsertEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['hand-insert-v2-goal-hidden']
# DrawerCloseEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['drawer-close-v2-goal-hidden']
# DrawerOpenEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['drawer-open-v2-goal-hidden']
# FaucetOpenEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['faucet-open-v2-goal-hidden']
# FaucetCloseEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['faucet-close-v2-goal-hidden']
# HammerEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['hammer-v2-goal-hidden']
# HandlePressSideEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['handle-press-side-v2-goal-hidden']
# HandlePressEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['handle-press-v2-goal-hidden']
# HandlePullSideEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['handle-pull-side-v2-goal-hidden']
# HandlePullEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['handle-pull-v2-goal-hidden']
# LeverPullEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['lever-pull-v2-goal-hidden']
# PegInsertSideEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['peg-insert-side-v2-goal-hidden']
# PickPlaceWallEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['pick-place-wall-v2-goal-hidden']
# PickOutOfHoleEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['pick-out-of-hole-v2-goal-hidden']
# ReachEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['reach-v2-goal-hidden']
# PushBackEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['push-back-v2-goal-hidden']
# PushEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['push-v2-goal-hidden']
# PickPlaceEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['pick-place-v2-goal-hidden']
# PlateSlideEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['plate-slide-v2-goal-hidden']
# PlateSlideSideEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['plate-slide-side-v2-goal-hidden']
# PlateSlideBackEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['plate-slide-back-v2-goal-hidden']
# PlateSlideBackSideEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['plate-slide-back-side-v2-goal-hidden']
# PegUnplugSideEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['peg-unplug-side-v2-goal-hidden']
# SoccerEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['soccer-v2-goal-hidden']
# StickPushEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['stick-push-v2-goal-hidden']
# StickPullEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['stick-pull-v2-goal-hidden']
# PushWallEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['push-wall-v2-goal-hidden']
# ReachWallEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['reach-wall-v2-goal-hidden']
# ShelfPlaceEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['shelf-place-v2-goal-hidden']
# SweepIntoEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['sweep-into-v2-goal-hidden']
# SweepEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['sweep-v2-goal-hidden']
# WindowOpenEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['window-open-v2-goal-hidden']
# WindowCloseEnv = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN['window-close-v2-goal-hidden']

