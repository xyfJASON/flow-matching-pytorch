import numpy as np

import torch
import torch.nn as nn
from torch import Tensor


class FlowMatchingSampler:
    def __init__(self):
        pass

    @staticmethod
    def euler_step(model_output: Tensor, timestep: float, previous_timestep: float, sample: Tensor):
        return sample - model_output * (timestep - previous_timestep)

    @torch.no_grad()
    def sample_step(self, model: nn.Module, x: Tensor, t: float, t_prev: float):
        t_batch = torch.full((x.shape[0], ), t, dtype=torch.float, device=x.device)
        model_output = model(x, t_batch)
        x = self.euler_step(model_output, t, t_prev, x)
        return x

    def sample_loop(self, model: nn.Module, init_noise: Tensor, sampling_steps: int):
        x = init_noise
        timesteps = np.linspace(1, 0, sampling_steps + 1)
        for i in range(sampling_steps):
            t, t_prev = timesteps[i], timesteps[i + 1]
            x = self.sample_step(model, x, t, t_prev)  # type: ignore
            yield x

    def sample(self, model: nn.Module, init_noise: Tensor, sampling_steps: int):
        *_, x = self.sample_loop(model, init_noise, sampling_steps)
        return x
