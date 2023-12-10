import copy
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
# import wandb


class BaseAlgorithm(ABC):
    def __init__(self, **kwargs):
        self._constructor_kwargs = copy.deepcopy(kwargs)
        self.__dict__.update(kwargs)
        # wandb.config = {k: v for k, v in kwargs.items() if isinstance(v, (int, float, bool, str))}

    @classmethod
    def load(cls, ckpt_path):
        ckpt = torch.load(ckpt_path)
        kwargs, state_dict = ckpt['constructor_kwargs'], ckpt['state_dict']
        algo = cls(**kwargs)
        algo.load_state_dict(state_dict)
        return algo

    def save(self, ckpt_path):
        ckpt = dict(
            constructor_kwargs=self._constructor_kwargs,
            state_dict=self._get_state_dict()
        )
        torch.save(ckpt, ckpt_path)

    def load_state_dict(self, state_dict: Dict):
        """
        Load internal states of a saved algorithm that are not directly recreated upon re-initialization.
        :param state_dict: a dictionary of {k: v} pairs where each v can be use to initialize a class attribute k
        """
        for k, v in state_dict.items():
            if k in self.__dict__:
                if isinstance(self.__dict__[k], BaseAlgorithm):
                    self.__dict__[k].load_state_dict(v)
                elif isinstance(self.__dict__[k], (nn.Module, nn.ModuleList, nn.ModuleDict, torch.optim.Optimizer)):
                    self.__dict__[k].load_state_dict(v)
                elif isinstance(self.__dict__[k], (int, float, bool, str, np.ndarray, nn.Parameter)):
                    self.__dict__[k] = v
                else:
                    raise NotImplementedError('Class attribute {} has type {} whose state_dict loading '
                                              'rule is not implemented'.format(k, type(self.__dict__[k])))
            else:
                raise KeyError('Class has no attribute {} that appears in state_dict. '
                               'Consider initialize the attribute in __init__ method'.format(k))

    @property
    @abstractmethod
    def _state_attributes(self):
        """
        Return a list of attribute names to store in state_dict.
        :return: a list of strings
        """
        raise NotImplementedError

    def _get_state_dict(self):
        state_dict = dict()
        for k, v in self.__dict__.items():
            if k in self._state_attributes:
                if isinstance(v, BaseAlgorithm):
                    state_dict[k] = v._get_state_dict()
                if isinstance(v, (nn.Module, nn.ModuleList, nn.ModuleDict, torch.optim.Optimizer)):
                    state_dict[k] = v.state_dict()
                elif isinstance(v, (int, float, bool, str, np.ndarray, nn.Parameter)):
                    state_dict[k] = v

        return state_dict



