"""Various 2-dim datasets."""

# From https://github.com/bayesiains/nsf/blob/master/data/plane.py

import numpy as np
import os
import torch

# from skimage import color, io, transform
# from torch import distributions
from torch.utils.data import Dataset


# import nsf_utils as utils

def load_plane_dataset(name, num_points, flip_axes=False, scale=True, npad=0):
    """Loads and returns a plane dataset.
    Args:
        name: string, the name of the dataset.
        num_points: int, the number of points the dataset should have,
        flip_axes: bool, flip x and y axes if True.
    Returns:
        A Dataset object, the requested dataset.
    Raises:
         ValueError: If `name` an unknown dataset.
    """

    try:
        dataset = {
            'gaussian': GaussianDataset,
            'crescent': CrescentDataset,
            'crescent_cubed': CrescentCubedDataset,
            'sine_wave': SineWaveDataset,
            'abs': AbsDataset,
            'sign': SignDataset,
            'four_circles': FourCircles,
            'diamond': DiamondDataset,
            'two_spirals': TwoSpiralsDataset,
            'checkerboard': CheckerboardDataset,
            'corners': CornersDataset,
            'eightgauss': EightGaussiansDataset,
            'checkerboard_modes': CheckerboardModes,

        }[name](num_points=num_points, flip_axes=flip_axes)
        if scale:
            # Scale data to be between zero and one
            # dataset.data = 2 * (dataset.data - dataset.data.min()) / (dataset.data.max() - dataset.data.min()) - 1
            dataset.data = (dataset.data + 4) / 4 - 1
        if npad > 0:
            padder = torch.distributions.uniform.Uniform(torch.zeros(npad), torch.ones(npad), validate_args=None)
            pads = padder.sample([num_points])
            dataset.data = torch.cat((dataset.data, pads), 1)
        return dataset

    except KeyError:
        raise ValueError('Unknown dataset: {}'.format(name))


class PlaneDataset(Dataset):
    def __init__(self, num_points, flip_axes=False):
        self.num_points = num_points
        self.flip_axes = flip_axes
        self.data = None
        self.reset()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def reset(self):
        self._create_data()
        if self.flip_axes:
            x1 = self.data[:, 0]
            x2 = self.data[:, 1]
            self.data = torch.stack([x2, x1]).t()

    def _create_data(self):
        raise NotImplementedError


class GaussianDataset(PlaneDataset):
    def _create_data(self):
        self.bounded = False
        x1 = torch.randn(self.num_points)
        x2 = 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x1, x2)).t()


class CrescentDataset(PlaneDataset):
    def _create_data(self):
        self.bounded = False
        x1 = torch.randn(self.num_points)
        x2_mean = 0.5 * x1 ** 2 - 1
        x2_var = torch.exp(torch.Tensor([-2]))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x2, x1)).t()


class CrescentCubedDataset(PlaneDataset):
    def _create_data(self):
        self.bounded = False
        x1 = torch.randn(self.num_points)
        x2_mean = 0.2 * x1 ** 3
        x2_var = torch.ones(x1.shape)
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x2, x1)).t()


class SineWaveDataset(PlaneDataset):
    def _create_data(self):
        self.bounded = False
        x1 = torch.randn(self.num_points)
        x2_mean = torch.sin(5 * x1)
        x2_var = torch.exp(-2 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x1, x2)).t()


class AbsDataset(PlaneDataset):
    def _create_data(self):
        self.bounded = False
        x1 = torch.randn(self.num_points)
        x2_mean = torch.abs(x1) - 1.
        x2_var = torch.exp(-3 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x1, x2)).t()


class SignDataset(PlaneDataset):
    def _create_data(self):
        self.bounded = False
        x1 = torch.randn(self.num_points)
        x2_mean = torch.sign(x1) + x1
        x2_var = torch.exp(-3 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x1, x2)).t()


class FourCircles(PlaneDataset):
    def __init__(self, num_points, flip_axes=False):
        if num_points % 4 != 0:
            raise ValueError('Number of data points must be a multiple of four')
        super().__init__(num_points, flip_axes)
        self.bounded = True

    @staticmethod
    def create_circle(num_per_circle, std=0.1):
        u = torch.rand(num_per_circle)
        x1 = torch.cos(2 * np.pi * u)
        x2 = torch.sin(2 * np.pi * u)
        data = 2 * torch.stack((x1, x2)).t()
        data += std * torch.randn(data.shape)
        return data

    def _create_data(self):
        num_per_circle = self.num_points // 4
        centers = [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ]
        self.data = torch.cat(
            [self.create_circle(num_per_circle) - torch.Tensor(center)
             for center in centers]
        )


class DiamondDataset(PlaneDataset):
    def __init__(self, num_points, flip_axes=False, width=20, bound=2.5, std=0.04):
        # original values: width=15, bound=2, std=0.05
        self.width = width
        self.bound = bound
        self.std = std
        super().__init__(num_points, flip_axes)
        self.bounded = True

    def _create_data(self, rotate=True):
        # probs = (1 / self.width**2) * torch.ones(self.width**2)
        #
        # means = torch.Tensor([
        #     (x, y)
        #     for x in torch.linspace(-self.bound, self.bound, self.width)
        #     for y in torch.linspace(-self.bound, self.bound, self.width)
        # ])
        #
        # covariance = self.std**2 * torch.eye(2)
        # covariances = covariance[None, ...].repeat(self.width**2, 1, 1)
        #
        # mixture_distribution = distributions.OneHotCategorical(
        #     probs=probs
        # )
        # components_distribution = distributions.MultivariateNormal(
        #     loc=means,
        #     covariance_matrix=covariances
        # )
        #
        # mask = mixture_distribution.sample((self.num_points,))[..., None].repeat(1, 1, 2)
        # samples = components_distribution.sample((self.num_points,))
        # self.data = torch.sum(mask * samples, dim=-2)
        # if rotate:
        #     rotation_matrix = torch.Tensor([
        #         [1 / np.sqrt(2), -1 / np.sqrt(2)],
        #         [1 / np.sqrt(2), 1 / np.sqrt(2)]
        #     ])
        #     self.data = self.data @ rotation_matrix
        means = np.array([
            (x + 1e-3 * np.random.rand(), y + 1e-3 * np.random.rand())
            for x in np.linspace(-self.bound, self.bound, self.width)
            for y in np.linspace(-self.bound, self.bound, self.width)
        ])

        covariance_factor = self.std * np.eye(2)

        index = np.random.choice(range(self.width ** 2), size=self.num_points, replace=True)
        noise = np.random.randn(self.num_points, 2)
        self.data = means[index] + noise @ covariance_factor
        if rotate:
            rotation_matrix = np.array([
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
                [1 / np.sqrt(2), 1 / np.sqrt(2)]
            ])
            self.data = self.data @ rotation_matrix
        self.data = self.data.astype(np.float32)
        self.data = torch.Tensor(self.data)


class TwoSpiralsDataset(PlaneDataset):
    def _create_data(self):
        self.bounded = True
        n = torch.sqrt(torch.rand(self.num_points // 2)) * 540 * (2 * np.pi) / 360
        d1x = -torch.cos(n) * n + torch.rand(self.num_points // 2) * 0.5
        d1y = torch.sin(n) * n + torch.rand(self.num_points // 2) * 0.5
        x = torch.cat([torch.stack([d1x, d1y]).t(), torch.stack([-d1x, -d1y]).t()])
        self.data = x / 3 + torch.randn_like(x) * 0.1


class TestGridDataset(PlaneDataset):
    def __init__(self, num_points_per_axis, bounds):
        self.num_points_per_axis = num_points_per_axis
        self.bounds = bounds
        self.shape = [num_points_per_axis] * 2
        self.X = None
        self.Y = None
        super().__init__(num_points=num_points_per_axis ** 2)

    def _create_data(self):
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], self.num_points_per_axis)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], self.num_points_per_axis)
        self.X, self.Y = np.meshgrid(x, y)
        data_ = np.vstack([self.X.flatten(), self.Y.flatten()]).T
        self.data = torch.tensor(data_).float()


class CheckerboardDataset(PlaneDataset):
    def _create_data(self):
        self.bounded = True
        x1 = torch.rand(self.num_points) * 4 - 2
        x2_ = torch.rand(self.num_points) - torch.randint(0, 2, [self.num_points]).float() * 2
        x2 = x2_ + torch.floor(x1) % 2
        self.data = torch.stack([x1, x2]).t() * 2

class RotatedCheckerboardDataset(PlaneDataset):
    def _create_data(self):
        self.bounded = True
        x1 = torch.rand(self.num_points) * 4 - 2
        x2_ = torch.rand(self.num_points) - torch.randint(0, 2, [self.num_points]).float() * 2
        x2 = x2_ + torch.floor(x1) % 2
        self.data = torch.stack([x2, x1]).t() * 2
        rotation_matrix = np.array([
            [0, -1],
            [1, 0]
        ], dtype=np.float32)
        self.data = self.data @ rotation_matrix

class CheckerboardModes(PlaneDataset):
    def __init__(self, num_points, flip_axes=False, epsilon=0.1):
        self.epsilon = epsilon
        super().__init__(num_points, flip_axes)
        self.bounded = True

    def _create_data(self):
        # Not really concerned with getting the probability exactly right...
        n_checkers = int(self.num_points * (1 - self.epsilon))
        in_dist = CheckerboardDataset(n_checkers).data
        n_ood = self.num_points - n_checkers
        oo_dist = RotatedCheckerboardDataset(n_ood).data
        idx = torch.randperm(self.num_points)
        self.data = torch.cat((in_dist, oo_dist))[idx]



# class FaceDataset(PlaneDataset):
#     def __init__(self, num_points, name='einstein', flip_axes=False):
#         self.name = name
#         self.image = None
#         super().__init__(num_points, flip_axes)
#         self.bounded = True
#
#     def _create_data(self):
#         root = utils.get_data_root()
#         path = os.path.join(root, 'faces', self.name + '.jpg')
#         try:
#             image = io.imread(path)
#         except FileNotFoundError:
#             raise RuntimeError('Unknown face name: {}'.format(self.name))
#         image = color.rgb2gray(image)
#         self.image = transform.resize(image, [512, 512])
#
#         grid = np.array([
#             (x, y) for x in range(self.image.shape[0]) for y in range(self.image.shape[1])
#         ])
#
#         rotation_matrix = np.array([
#             [0, -1],
#             [1, 0]
#         ])
#         p = self.image.reshape(-1) / sum(self.image.reshape(-1))
#         ix = np.random.choice(range(len(grid)), size=self.num_points, replace=True, p=p)
#         points = grid[ix].astype(np.float32)
#         points += np.random.rand(self.num_points, 2)  # dequantize
#         points /= (self.image.shape[0])  # scale to [0, 1]
#         # assert 0 <= min(points) <= max(points) <= 1
#
#         self.data = torch.tensor(points @ rotation_matrix).float()
#         self.data[:, 1] += 1


# From https://github.com/didriknielsen/survae_flows/blob/271a31d11cf00fad83270a9f699ff68a03da44ab/experiments/toy/datasets.py
class CornersDataset(PlaneDataset):
    '''Adapted from https://www.mathworks.com/matlabcentral/fileexchange/41459-6-functions-for-generating-artificial-datasets'''

    def _create_data(self):
        self.bounded = True
        assert self.num_points % 8 == 0
        N = self.num_points
        scale = 1
        gapwidth = 1
        cornerwidth = 3

        xplusmin = torch.cat([torch.ones(N // 4), -torch.ones(N // 4), torch.ones(N // 4), -torch.ones(N // 4)])
        yplusmin = torch.cat([torch.ones(N // 4), -torch.ones(N // 2), torch.ones(N // 4)])

        horizontal = torch.cat([xplusmin[::2] * gapwidth + xplusmin[::2] * scale * torch.rand(N // 2),
                                yplusmin[::2] * gapwidth + cornerwidth * yplusmin[::2] * torch.rand(N // 2)], dim=0)

        vertical = torch.cat([xplusmin[1::2] * gapwidth + cornerwidth * xplusmin[1::2] * torch.rand(N // 2),
                              yplusmin[1::2] * gapwidth + yplusmin[1::2] * scale * torch.rand(N // 2)], dim=0)

        data = torch.stack([horizontal, vertical], dim=-1)
        data[..., 0] *= (2 * torch.bernoulli(0.5 * torch.ones(N)) - 1)
        data[..., 1] *= (2 * torch.bernoulli(0.5 * torch.ones(N)) - 1)

        self.data = data


class EightGaussiansDataset(PlaneDataset):
    '''Adapted from https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py'''

    def _create_data(self):
        self.bounded = False
        scale = 4.
        bias = np.pi / 8
        step = np.pi / 4
        centers = [(np.cos(bias + 0 * step), np.sin(bias + 0 * step)),
                   (np.cos(bias + 1 * step), np.sin(bias + 1 * step)),
                   (np.cos(bias + 2 * step), np.sin(bias + 2 * step)),
                   (np.cos(bias + 3 * step), np.sin(bias + 3 * step)),
                   (np.cos(bias + 4 * step), np.sin(bias + 4 * step)),
                   (np.cos(bias + 5 * step), np.sin(bias + 5 * step)),
                   (np.cos(bias + 6 * step), np.sin(bias + 6 * step)),
                   (np.cos(bias + 7 * step), np.sin(bias + 7 * step))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(self.num_points):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        self.data = torch.from_numpy(dataset)


def _test():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset = DiamondDataset(num_points=int(1e6), width=20, bound=2.5, std=0.04)

    from nsf_utils import torchutils
    from matplotlib import pyplot as plt
    data = torchutils.tensor2numpy(dataset.data)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.scatter(data[:, 0], data[:, 1], s=2, alpha=0.5)
    bound = 4
    bounds = [[-bound, bound], [-bound, bound]]
    # bounds = [
    #     [0, 1],
    #     [0, 1]
    # ]
    ax.hist2d(data[:, 0], data[:, 1], bins=256, range=bounds)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    plt.show()


if __name__ == '__main__':
    _test()
