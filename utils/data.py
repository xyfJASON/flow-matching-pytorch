import math
import random
from PIL import Image

import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from datasets import FFHQ, ImageNet


def load_data(conf, split='train'):
    """Keys in conf: 'name', 'root', 'img_size'."""

    if conf.name.lower() == 'mnist':
        transform = T.Compose([
            T.Resize((conf.img_size, conf.img_size), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        dataset = dset.MNIST(root=conf.root, train=(split == 'train'), transform=transform)

    elif conf.name.lower() in ['cifar10', 'cifar-10']:
        flip_p = 0.5 if split == 'train' else 0.0
        transform = T.Compose([
            T.Resize((conf.img_size, conf.img_size), antialias=True),
            T.RandomHorizontalFlip(p=flip_p),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        dataset = dset.CIFAR10(root=conf.root, train=(split == 'train'), transform=transform)

    elif conf.name.lower() == 'celeba':
        # https://github.com/NVlabs/stylegan/blob/master/dataset_tool.py#L484-L499
        flip_p = 0.5 if split in ['train', 'all'] else 0.0
        cx, cy = 89, 121
        transform = T.Compose([
            T.Lambda(lambda x: TF.crop(x, top=cy-64, left=cx-64, height=128, width=128)),
            T.Resize((conf.img_size, conf.img_size), antialias=True),
            T.RandomHorizontalFlip(flip_p),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = dset.CelebA(root=conf.root, split=split, transform=transform)

    elif conf.name.lower() == 'ffhq':
        flip_p = 0.5 if split in ['train', 'all'] else 0.0
        transform = T.Compose([
            T.Resize((conf.img_size, conf.img_size), antialias=True),
            T.RandomHorizontalFlip(p=flip_p),
            T.ToTensor(),
            T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])
        dataset = FFHQ(root=conf.root, split=split, transforms=transform)

    elif conf.name.lower() == 'imagenet':
        flip_p = 0.5 if split == 'train' else 0.0
        if split == 'train':
            crop = getattr(conf, 'crop', 'center')
            if crop == 'center':
                crop_arr = center_crop_arr
            elif crop == 'random':
                crop_arr = random_crop_arr
            else:
                raise NotImplementedError(f'Unknown crop: {crop}')
        else:
            crop_arr = center_crop_arr
        transform = T.Compose([
            T.Lambda(lambda pil_image: crop_arr(pil_image, conf.img_size)),
            T.RandomHorizontalFlip(p=flip_p),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        dataset = ImageNet(root=conf.root, split=split, transforms=transform)

    else:
        raise NotImplementedError(f'Unknown dataset: {conf.name}')

    return dataset


def center_crop_arr(pil_image, image_size):
    """Resize and center cropping from ADM, also used in DiT, LlamaGen, MAR, etc.

    References:
      - https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py#L126-L167
      - https://github.com/facebookresearch/DiT/blob/main/train.py#L85-L103
      - https://github.com/FoundationVision/LlamaGen/blob/main/dataset/augmentation.py
      - https://github.com/LTH14/mar/blob/main/util/crop.py

    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """Resize and random cropping from ADM, also used in LlamaGen, etc.

    References:
      - https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py#L126-L167
      - https://github.com/FoundationVision/LlamaGen/blob/main/dataset/augmentation.py

    """
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
