from .base import (
    get_uci_dataset_range,
    get_uci_dataset_max_abs_value,
    load_dataset,
    batch_generator,
    InfiniteLoader,
    load_num_batches,
    UnlabelledImageFolder
)


from .celeba import CelebA, CelebAHQ, CelebAHQ64Fast

from .cifar10 import CIFAR10Fast

from .imagenet import ImageNet32, ImageNet64, ImageNet64Fast