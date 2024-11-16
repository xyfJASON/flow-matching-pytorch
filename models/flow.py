import numpy as np

import torch
import torch.nn as nn
from torch import Tensor


class FlowMatchingSampler:
    def __init__(self, method: str = 'euler'):
        if method not in ['euler', 'heun']:
            raise ValueError(f'Method {method} not supported.')
        self.method = method

    @torch.no_grad()
    def euler_step(self, model: nn.Module, x: Tensor, t: float, t_prev: float):
        t_batch = torch.full((x.shape[0], ), t, dtype=torch.float, device=x.device)
        model_output = model(x, t_batch * 999)
        x = x - model_output * (t - t_prev)
        return x

    @torch.no_grad()
    def heun_step(self, model: nn.Module, x: Tensor, t: float, t_prev: float):
        dt = t - t_prev
        t_batch = torch.full((x.shape[0], ), t, dtype=torch.float, device=x.device)
        dt_batch = torch.full((x.shape[0], ), dt, dtype=torch.float, device=x.device)
        k1 = model(x, t_batch * 999)
        k2 = model(x - k1 * dt, (t_batch - dt_batch) * 999)
        x = x - 0.5 * (k1 + k2) * dt
        return x

    def sample_loop(self, model: nn.Module, init_noise: Tensor, sampling_steps: int):
        x = init_noise
        timesteps = np.linspace(1, 0, sampling_steps + 1)
        for i in range(sampling_steps):
            t = timesteps[i].item()
            t_prev = timesteps[i + 1].item()
            if self.method == 'euler':
                x = self.euler_step(model, x, t, t_prev)
            elif self.method == 'heun':
                x = self.heun_step(model, x, t, t_prev)
            yield x

    def sample(self, model: nn.Module, init_noise: Tensor, sampling_steps: int):
        *_, x = self.sample_loop(model, init_noise, sampling_steps)
        return x
