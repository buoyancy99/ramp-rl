import numpy as np
import torch
import torch.nn as nn
from itertools import zip_longest


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def polyak_update(
    params,
    target_params,
    tau,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93

    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def zip_strict(*iterables):
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


class OnlineLeastSquare:
    def __init__(self):
        # use Sherman–Morrison formula
        self.AtA_inv = None
        self.AtB = None
        self.latest_w = None
        self.initialized = False
        self.update_rate = 1.0

    def initialize(self, A, B):
        A = A.astype(np.float64)
        B = B.astype(np.float64)
        self.AtA_inv = np.linalg.inv(A.T @ A)
        self.AtB = A.T @ B
        self.latest_w = self.AtA_inv @ self.AtB
        self.initialized = True

    def update(self, a, b):
        assert self.initialized
        a, b = a * self.update_rate, b * self.update_rate
        b = np.array(b)
        if a.ndim == 1:
            a = a[None]
        if b.ndim == 0:
            b = b[None]
        k = a.shape[0]
        C = self.AtA_inv @ a.T
        self.AtA_inv = self.AtA_inv - C @ np.linalg.inv(np.eye(k) + a @ C) @ C.T
        self.AtB = self.AtB + a.T @ b
        self.latest_w = self.AtA_inv @ self.AtB

        return self.latest_w

    def calculate(self):
        return self.latest_w

    def adjust_update_rate(self, rate):
        self.update_rate = rate


if __name__ == '__main__':
    ols = OnlineLeastSquare()
    A = np.random.randn(5000, 2800) * 10 - 5
    w_gt = np.random.rand(2800) * 2 - 1
    noise_gt = np.random.randn(5000)
    b = A @ w_gt + noise_gt
    ols.initialize(A[:3000], b[:3000])
    for i in range(3000, 5000):
        w_online = ols.update(A[i], b[i])

    online_loss = np.linalg.norm(A @ w_online - b)

    w_offline = np.linalg.lstsq(A, b, rcond=None)[0]
    offline_loss = np.linalg.norm(A @ w_offline - b)
    gt_loss = np.linalg.norm(noise_gt)
    print(offline_loss, online_loss, gt_loss)