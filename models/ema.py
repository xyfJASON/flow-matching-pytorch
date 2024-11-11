from contextlib import contextmanager
from typing import Iterable, Union

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of model parameters.

    References:
        - https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py
        - https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/utils.py
        - https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py
        - https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/ema.py
        - https://github.com/lucidrains/ema-pytorch
    """

    def __init__(
            self,
            parameters: Iterable[nn.Parameter],
            decay: float = 0.9999,
            ema_warmup_type: str = 'none',
            inv_gamma: Union[float, int] = 1.0,
            power: Union[float, int] = 2 / 3,
    ):
        """
        Args:
            parameters: The parameters to track, typically from `model.parameters()`.
            decay: The decay factor for exponential moving average.
            ema_warmup_type: Type of EMA warmup. Options: 'none', 'tensorflow-like', 'crowsonkb'.
            inv_gamma: Inverse multiplicative factor of EMA warmup. Only used when `ema_warmup_type` is 'crowsonkb'.
            power: Exponential factor of EMA warmup. Only used when `ema_warmup_type` is 'crowsonkb'.

        @crowsonkb's notes on EMA Warmup:
            If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
            good values for models you plan to train for a million or more steps (reaches decay
            factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
            you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
            215.4k steps).
        """
        assert ema_warmup_type in ['none', 'tensorflow-like', 'crowsonkb']

        self.decay = decay
        self.ema_warmup_type = ema_warmup_type
        self.inv_gamma = inv_gamma
        self.power = power

        self.num_updates = 0
        self.shadow_params = [param.clone().detach() for param in parameters]
        self.backup_params = []

    def get_decay(self, num_updates: int):
        if self.ema_warmup_type == 'none':
            decay = self.decay
        elif self.ema_warmup_type == 'tensorflow-like':
            decay = (1 + num_updates) / (10 + num_updates)
        elif self.ema_warmup_type == 'crowsonkb':
            decay = 1 - (1 + num_updates / self.inv_gamma) ** -self.power
        else:
            raise ValueError(f'Unknown EMA warmup type: {self.ema_warmup_type}')
        decay = min(self.decay, decay)
        return decay

    @torch.no_grad()
    def update(self, parameters: Iterable[nn.Parameter]):
        """Update the EMA parameters."""
        self.num_updates += 1
        decay = self.get_decay(self.num_updates)
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_((1. - decay) * (s_param - param))
            else:
                s_param.copy_(param)

    def apply(self, parameters: Iterable[nn.Parameter]):
        """Backup the original parameters and apply the EMA parameters."""
        assert len(self.backup_params) == 0, 'backup_params is not empty'
        for s_param, param in zip(self.shadow_params, parameters):
            self.backup_params.append(param.detach().cpu().clone())
            param.data.copy_(s_param.data)

    def restore(self, parameters: Iterable[nn.Parameter]):
        """Restore the original parameters from the backup."""
        assert len(self.backup_params) > 0, 'backup_params is empty'
        for b_param, param in zip(self.backup_params, parameters):
            param.data.copy_(b_param.to(param.device).data)
        self.backup_params = []

    @contextmanager
    def scope(self, parameters: Iterable[nn.Parameter]):
        """A context manager to apply and restore the EMA parameters."""
        parameters = list(parameters)
        self.apply(parameters)
        yield
        self.restore(parameters)

    def state_dict(self):
        return dict(
            decay=self.decay,
            shadow_params=self.shadow_params,
            num_updates=self.num_updates,
        )

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow_params = state_dict['shadow_params']
        self.num_updates = state_dict['num_updates']

    def to(self, device):
        self.shadow_params = [s_param.to(device) for s_param in self.shadow_params]
