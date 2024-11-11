import os
from typing import Sequence

import torch
from torch import Tensor
from torchvision.utils import save_image


def check_range(image: Tensor, low: float, high: float):
    return torch.ge(image, low).all() and torch.le(image, high).all()


def image_float_to_uint8(image: Tensor):
    """ [0, 1] -> [0, 255] """
    assert image.dtype == torch.float32
    assert check_range(image, 0, 1)
    return (image * 255).to(dtype=torch.uint8)


def image_norm_to_float(image: Tensor):
    """ [-1, 1] -> [0, 1] """
    assert image.dtype == torch.float32
    assert check_range(image, -1, 1)
    return (image + 1) / 2


def image_norm_to_uint8(image: Tensor):
    """ [-1, 1] -> [0, 255] """
    assert image.dtype == torch.float32
    assert check_range(image, -1, 1)
    return ((image + 1) / 2 * 255).to(dtype=torch.uint8)


def save_images(images: Sequence[Tensor], save_dir: str, start_idx: int):
    for i, image in enumerate(images):
        assert check_range(image, 0, 1)
        save_image(image, os.path.join(save_dir, f'{start_idx + i}.png'))
