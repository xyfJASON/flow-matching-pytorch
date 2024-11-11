"""Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/optimization.py"""

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class ConstantLR(LambdaLR):
    """Constant learning rate scheduler."""
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        super().__init__(optimizer, lambda _: 1.0, last_epoch)


class ConstantWarmupLR(LambdaLR):
    """Constant learning rate scheduler with warmup."""
    def __init__(self, optimizer: Optimizer, warmup_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1.0, self.warmup_steps))
        return 1.0


class LinearWarmupLR(LambdaLR):
    """Linear learning rate scheduler with warmup."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, training_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.training_steps - current_step) /
            float(max(1, self.training_steps - self.warmup_steps))
        )


class CosineWarmupLR(LambdaLR):
    """Cosine learning rate scheduler with warmup."""
    def __init__(self, optimizer: Optimizer, warmup_steps: int, training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.num_cycles = num_cycles
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        progress = float(current_step - self.warmup_steps) / float(max(1, self.training_steps - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))


class PolynomialWarmupLR(LambdaLR):
    """Polynomial learning rate scheduler with warmup."""
    def __init__(self, optimizer: Optimizer, warmup_steps: int, training_steps: int, lr_end: float = 1e-7, power: float = 1.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.lr_end = lr_end
        self.power = power
        self.lr_init = optimizer.defaults["lr"]
        if not (self.lr_init > self.lr_end):
            raise ValueError(f"lr_end ({self.lr_end}) must be be smaller than initial lr ({self.lr_init})")
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        elif current_step > self.training_steps:
            return self.lr_end / self.lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = self.lr_init - self.lr_end
            decay_steps = self.training_steps - self.warmup_steps
            pct_remaining = 1 - (current_step - self.warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**self.power + self.lr_end
            return decay / self.lr_init  # as LambdaLR multiplies by lr_init
