import argparse
import os

from torch.utils.data import DataLoader

from funnels.data.image_data import get_image_data, Preprocess
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar-10', help='String name of the dataset')
    parser.add_argument('--num_bits', type=int, default=5, help='Number of bits')
    return parser.parse_args()


def dump_images(dataset, num_bits):
    if dataset != 'mnist':
        save_dataset, _ = get_image_data(dataset, num_bits, train=False)
    else:
        save_dataset, _, _ = get_image_data(dataset, num_bits, train=True, valid_frac=0.01)
    save_loader = DataLoader(dataset=save_dataset,
                             batch_size=1000,
                             num_workers=0)
    top_dir = f'/scratch/{dataset}_{num_bits}'
    os.makedirs(top_dir, exist_ok=True)
    for j, data in enumerate(save_loader):
        [save_image(Preprocess(num_bits).inverse(image), f'{top_dir}/_{i}_{j}.jpg') for i, image in
         enumerate(data[0])]


if __name__ == '__main__':
    args = parse_args()
    dump_images(args.dataset, args.num_bits)
