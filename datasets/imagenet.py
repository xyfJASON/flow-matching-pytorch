import os
from PIL import Image
from typing import Optional, Callable

from torchvision.datasets import VisionDataset


def extract_images(root):
    """ Extract all images under root """
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    root = os.path.expanduser(root)
    img_paths = []
    for curdir, subdirs, files in os.walk(root):
        for file in files:
            if os.path.splitext(file)[1].lower() in img_ext:
                img_paths.append(os.path.join(curdir, file))
    img_paths = sorted(img_paths)
    return img_paths


class ImageNet(VisionDataset):
    """The ImageNet-1K (ILSVRC 2012) Dataset.

    Please organize the dataset in the following file structure:

    root
    ├── train
    │   ├── n01440764
    │   ├── ...
    │   └── n15075141
    ├── val
    │   ├── n01440764
    │   ├── ...
    │   └── n15075141
    └── test
        ├── ILSVRC2012_test_00000001.JPEG
        ├── ...
        └── ILSVRC2012_test_00100000.JPEG

    References:
      - https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
      - https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4
      - https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

    """

    def __init__(self, root: str, split: str = 'train', transforms: Optional[Callable] = None):
        super().__init__(root=root, transforms=transforms)

        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Invalid split: {split}')
        self.split = split

        # Extract image paths
        image_root = os.path.join(self.root, split if split != 'valid' else 'val')
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')
        self.img_paths = extract_images(image_root)

        # Extract class labels
        self.classes = None
        if self.split != 'test':
            class_names = [path.split('/')[-2] for path in self.img_paths]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            self.classes = [sorted_classes[x] for x in class_names]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int):
        x = Image.open(self.img_paths[index]).convert('RGB')
        y = self.classes[index] if self.classes is not None else None
        if self.transforms is not None:
            x = self.transforms(x)
        if y is None:
            return x
        return x, y
