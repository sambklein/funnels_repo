import numpy as np
import os
import torch
from itertools import combinations_with_replacement, permutations

from torch.utils.data import Dataset

# TODO: rename this, subclasses are not plane datasets
class HyperPlaneDataset(Dataset):
    def __init__(self, num_points, dim, flip_axes=False):
        self.num_points = num_points
        self.dim = dim
        self.flip_axes = flip_axes
        self.data = None
        self._create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def _create_data(self):
        raise NotImplementedError

    def sample(self, num):
        # Expect a list for the num
        self.num_points = num[0]
        self._create_data()
        return self.data


class SparseHyperCheckerboardDataset(HyperPlaneDataset):
    """
    This class generates an object that has a checkerboard in every consecutive 2D projection. It is sparse in the sense
    that the number of hyper checkers it contains is sparse - the density per hyper checker will be higher than the
    class below. With each extra dimension added the number of checkers doubles.
    """

    def __init__(self, num_points, dim, flip_axes=False):
        super(SparseHyperCheckerboardDataset, self).__init__(num_points, dim, flip_axes=flip_axes)

    def append_axis(self, x):
        xi_ = torch.rand(self.num_points) - torch.randint(0, 2, [self.num_points]).float() * 2
        return xi_ + torch.floor(x) % 2

    def _create_data(self):
        self.bounded = True
        if self.dim > 1:
            x1 = torch.rand(self.num_points) * 4 - 2
            axis_list = [x1]
            for i in range(1, self.dim):
                axis_list += [self.append_axis(axis_list[-1])]
            self.data = torch.stack(axis_list).t() * 2
        else:
            self.data = torch.cat(
                (torch.rand(int(self.num_points / 2)) * 2 - 4, torch.rand(int(self.num_points / 2)) * 2))


class HyperCheckerboardDataset(HyperPlaneDataset):
    """
    This class generates an object that is truly a checkerboard. With each dimension that is added the number of
    checkers qudruples.
    """

    def __init__(self, num_points, dim, flip_axes=False):
        super(HyperCheckerboardDataset, self).__init__(num_points, dim, flip_axes=flip_axes)

    def make_cube(self):
        return torch.rand(self.dim, self.num_points)

    @staticmethod
    def count_oob(cube):
        """
        Get the fraction of samples outside of the bounds of the cube
        """
        out_range = (cube > 4).any(1) | (cube < -4).any(1)
        out_range = out_range.sum() / cube.shape[0]
        return out_range

    @staticmethod
    def mask_ood(cube):
        """
        :param cube: A tensor of samples from a cube.shape[1] dimensional space.
        :return: A mask with ones where samples are OOD
        """
        dim = cube.shape[1]
        # Get the cube assignments for each axis
        labels = ((cube + 4) / 2).floor() % 2
        # If the sum is odd and so is the dimension then the point is in the checkerboard, and the same for even
        mx = labels.sum(1) % (2 + dim % 2)
        # We also need to set all points outside of the data range to one in mx
        out_range = (cube > 4).any(1) | (cube < -4).any(1)
        return (out_range.type(mx.dtype) + mx) > 0

    @staticmethod
    def count_ood(cube):
        """
        :param cube: A tensor of samples from a cube.shape[1] dimensional space.
        :return: The fraction of samples that are within a hypercheckerboard
        """
        return HyperCheckerboardDataset.mask_ood(cube).sum() / cube.shape[0]

    @staticmethod
    def split_cube(cube, flip_axes=False):
        """
        An n-dimensional checkerboard is just a set of 2D checkerboards, therefore all that is required is to find the
        correct shift in a two dimensional plane. This is defined by a set of oscillating transformations depending on
        the value of the nth coordinates.
        :param cube: an n-dimensional cube with values uniformly distributed in (0, 1)
        :return: an n-dimensional checkerboard
        """
        # Split first axis
        ax0 = cube[0]
        ax0 -= 0.5
        ax0[ax0 < 0] = ax0[ax0 < 0] * 2 - 1
        ax0[ax0 > 0] = ax0[ax0 > 0] * 2
        if cube.shape[0] > 1:
            # Scale other axes to be in a useful range for floor divide
            cube[1:] = cube[1:] * 4
            # Define the shifts
            displace = cube[1:].floor() % 2
            shift = displace[0]
            # We need an algebra that satisies: 1 * 0 = 0, 1 * 1 = 1, 0 * 1 = 0, 0 * 0 = 1
            # This is achieved with * = (==)
            for ax in displace[1:]:
                shift = shift == ax
            ax0 += shift
            cube[1:] -= 2
        cube *= 2
        return cube.t()

    def _create_data(self):
        self.bounded = True
        # All checkerboards start from an N dim checkerboard in [0, 1]
        cube = self.make_cube()
        self.data = self.split_cube(cube, flip_axes=self.flip_axes)


class HyperNonUniformSphere(HyperPlaneDataset):
    def __init__(self, num_points, dim, flip_axes=False):
        super().__init__(num_points, dim, flip_axes)
        self.bounded = True

    def create_sphere(self, std=0.1):
        angles = np.pi * torch.rand((self.dim - 1, self.num_points))
        angles[-1] *= 2
        data = torch.ones((self.dim, self.num_points))
        for coord in range(self.dim - 1):
            j = coord + 1
            for i, u in enumerate(angles[:j]):
                if i == (j - 1):
                    data[coord] *= torch.cos(u)
                else:
                    data[coord] *= torch.sin(u)

        if angles.shape[0] == 1:
            data[-1] = torch.sin(angles)
        else:
            data[-1] = torch.prod(torch.sin(angles), 0)

        data = 2 * data.t()
        data += std * torch.randn(data.shape)
        return data

    def _create_data(self):
        self.data = self.create_sphere()

class HyperSphere(HyperPlaneDataset):
    def __init__(self, num_points, dim, flip_axes=False, radius=2, std=0.05):
        self.radius = radius
        self.std = std
        super().__init__(num_points, dim, flip_axes)
        self.bounded = False

    def create_sphere(self):
        vectors = torch.normal(0, 1, size=(self.num_points, self.dim))
        # Make a sphere of radius self.radius
        data = self.radius * vectors / (torch.sum(vectors ** 2, 1) ** 0.5).view(-1, 1)
        data += self.std * torch.randn(data.shape)
        return data

    def _create_data(self):
        self.data = self.create_sphere()


class HyperSouthernCross(HyperPlaneDataset):
    """Hyper southern cross."""

    def __init__(self, num_points, dim, flip_axes=False, width=20, bound=2.5, std=0.1):
        self.width = width
        self.bound = bound
        self.std = std
        rotation_matrix = np.array([
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
                [1 / np.sqrt(2), 1 / np.sqrt(2)]
            ])
        self.rotation_matrix = np.kron(np.eye(int(np.ceil(dim / 2)), dtype=int), rotation_matrix)
        self.rotation_matrix = self.rotation_matrix[:dim, :dim]
        super().__init__(num_points, dim, flip_axes)
        self.bounded = True

    def _create_data(self, rotate=True):
        loc = np.random.randint(0, 2, self.num_points) * 2 - 1
        n_uni = torch.randint(0, self.dim, (self.num_points, 1))
        inds = torch.cat((torch.arange(self.num_points).view(self.num_points, 1), n_uni), 1)
        self.data = np.zeros((self.num_points, self.dim))
        self.data[np.arange(self.num_points), inds[:, 1].numpy()] = loc

        covariance_factor = self.std * np.eye(self.dim)
        noise = np.random.randn(self.num_points, self.dim)
        self.data += noise @ covariance_factor
        if rotate:
            self.data = self.data @ self.rotation_matrix
        self.data = self.data.astype(np.float32)
        self.data = torch.Tensor(self.data)


class HyperCross(HyperPlaneDataset):
    def __init__(self, num_points, dim, flip_axes=False):
        self.q, r = torch.qr(torch.randn((dim, dim)))
        self.npoles = 4 if 4 < dim else dim
        super().__init__(num_points, dim, flip_axes)
        self.bounded = False

    def create_cross(self, std=0.5):
        scale = torch.sqrt(torch.rand(self.num_points)) * 4 - 2
        scale[scale < 0] += std
        scale[scale > 0] += std
        n_uni = torch.randint(1, self.dim, (self.num_points, 1))
        inds = torch.cat((torch.arange(self.num_points).view(self.num_points, 1), n_uni), 1)
        data = np.zeros((self.num_points, self.dim))
        data[np.arange(self.num_points), inds[:, 1].numpy()] = scale.numpy()
        data = torch.tensor(data, dtype=torch.float32)
        data = data.matmul(self.q)
        data += std * torch.rand(data.shape)
        return data

    def _create_data(self):
        self.data = self.create_cross()

class HyperCube(HyperPlaneDataset):
    def __init__(self, num_points, dim, flip_axes=False):
        self.q, r = torch.qr(torch.randn((dim, dim)))
        super().__init__(num_points, dim, flip_axes)
        self.bounded = False

    def create_cross(self, std=0.5):
        n_uni = torch.randint(1, self.dim, (self.num_points, 1))
        inds = torch.cat((torch.arange(self.num_points).view(self.num_points, 1), n_uni), 1)
        sign = 2 * np.random.randint(0, 2, size=(self.num_points)) - 1
        data = np.random.uniform(-1, 1, (self.num_points, self.dim))
        data[np.arange(self.num_points), inds[:, 1].numpy()] = 2 * sign
        data = torch.tensor(data, dtype=torch.float32)
        # data = data.matmul(self.q)
        data += std * torch.rand(data.shape)
        return data

    def _create_data(self):
        self.data = self.create_cross()


class NestedCubes(HyperPlaneDataset):
    """Nested cubes"""
    def __init__(self, num_points, dim, flip_axes=False):
        self.q, r = torch.qr(torch.randn((dim, dim)))
        self.q2, r = torch.qr(torch.randn((dim, dim)))
        super().__init__(num_points, dim, flip_axes)
        self.bounded = False

    def create_cube(self, num_points, std=0.1):
        n_uni = torch.randint(1, self.dim, (num_points, 1))
        inds = torch.cat((torch.arange(num_points).view(num_points, 1), n_uni), 1)
        sign = 2 * np.random.randint(0, 2, size=(num_points)) - 1
        data = np.random.uniform(-1, 1, (num_points, self.dim))
        data[np.arange(num_points), inds[:, 1].numpy()] = sign
        data = torch.tensor(data, dtype=torch.float32)
        # data = data.matmul(self.q)
        data += std * torch.rand(data.shape)
        return data

    def _create_data(self):
        num_per_cube = self.num_points // 2
        widths = [2, 1]
        rotations = [self.q, self.q2]
        self.data = torch.cat(
            [self.create_cube(num_per_cube).matmul(rot) * width for rot, width in zip(rotations, widths)]
        )


class HyperCubeGauss(HyperPlaneDataset):
    """Hyper cube with gaussians on the surface."""
    def __init__(self, num_points, dim, flip_axes=False):
        super().__init__(num_points, dim, flip_axes)
        self.bounded = False

    def create_cube(self, num_points, shift, std=0.1):
        n_uni = torch.randint(1, self.dim, (num_points, 1))
        inds = torch.cat((torch.arange(num_points).view(num_points, 1), n_uni), 1)
        clip = 3
        sign = 2 * np.random.randint(0, 2, size=(num_points)) - 1
        data = np.clip(np.random.normal(shift, 0.5, (num_points, self.dim)), -clip, clip)
        data[np.arange(num_points), inds[:, 1].numpy()] = sign * clip
        data = torch.tensor(data, dtype=torch.float32)
        # data = data.matmul(self.q)
        data += std * torch.rand(data.shape)
        return data

    def _create_data(self):
        num_per_cube = self.num_points // 2
        shifts = [-1, 1]

        self.data = torch.cat(
            [self.create_cube(num_per_cube, shift) for shift in shifts]
        )


class HyperDiamondRandom(HyperPlaneDataset):
    """Hyper diamond distribution random placements - doesn't work very well in low dimensions."""

    def __init__(self, num_points, dim, flip_axes=False, width=20, bound=2.5, std=0.04):
        # original values: width=15, bound=2, std=0.05
        self.width = width
        self.bound = bound
        self.std = std
        self.rotation_matrix, r = np.linalg.qr(np.random.standard_normal((dim, dim)))
        self.inds = np.random.randint(0, width, (width ** 2, dim))
        super().__init__(num_points, dim, flip_axes)
        self.bounded = True

    def _create_data(self, rotate=True):
        # means = np.array([
        #     (x + 1e-3 * np.random.rand(), y + 1e-3 * np.random.rand())
        #     for x in np.linspace(-self.bound, self.bound, self.width)
        #     for y in np.linspace(-self.bound, self.bound, self.width)
        # ])
        arrs = [np.linspace(-self.bound, self.bound, self.width)] * self.dim
        arrs = [arr + 1e-3 * np.random.standard_normal(self.width) for arr in arrs]
        grid = np.array(arrs)
        means = grid[np.arange(self.dim), self.inds]

        covariance_factor = self.std * np.eye(self.dim)

        index = np.random.choice(range(self.width ** 2), size=self.num_points, replace=True)
        noise = np.random.randn(self.num_points, self.dim)
        self.data = means[index] + noise @ covariance_factor
        if rotate:
            self.data = self.data @ self.rotation_matrix
        self.data = self.data.astype(np.float32)
        self.data = torch.Tensor(self.data)


class HyperShells(HyperPlaneDataset):
    def __init__(self, num_points, dim, flip_axes=False, std=0.1):
        self.bounded = False
        self.nspheres = 4
        self.std = std
        if num_points % self.nspheres != 0:
            raise ValueError(f'Number of data points must be a multiple of {self.nspheres}')
        self.radii = [0.5, 1, 1.5, 2]
        super().__init__(num_points, dim, flip_axes)

    def create_sphere(self, num_per_circle, radius):
        return HyperSphere(num_per_circle, self.dim, radius=radius, std=self.std).data

    def _create_data(self):
        num_per_circle = self.num_points // self.nspheres
        self.data = torch.cat(
            [self.create_sphere(num_per_circle, radius)
             for radius in self.radii]
        )

class HyperSpheres(HyperPlaneDataset):
    def __init__(self, num_points, dim, flip_axes=False):
        if dim % 2 != 0:
            raise ValueError('Dim must be a multiple of 2')
        self.nperm = 4  # dim
        angles = (2 * np.pi / self.nperm) * np.arange(self.nperm)
        centers = []
        for i in range(self.nperm):
            centers += [[np.cos(angles[i]), np.sin(angles[i])] * int(dim // 2)]
        self.radius = 0.5
        self.centers = torch.tensor(centers, dtype=torch.float32) * 2.5
        super().__init__(num_points, dim, flip_axes)
        self.bounded = False

    def create_sphere(self, num_per_circle, std=0.1):
        return HyperSphere(num_per_circle, self.dim).data

    def _create_data(self):
        num_per_circle = self.num_points // self.nperm
        self.data = torch.cat(
            [self.create_sphere(num_per_circle) * self.radius - torch.Tensor(center)
             for center in self.centers]
        )

class HyperCylinders(HyperPlaneDataset):
    def __init__(self, num_points, dim, flip_axes=False):
        if dim % 2 != 0:
            raise ValueError('Dim must be a multiple of 2')
        self.nperm = dim
        angles = (2 * np.pi / self.nperm) * np.arange(self.nperm)
        centers = []
        for i in range(self.nperm):
            centers += [[np.cos(angles[i]), np.sin(angles[i])] * int(self.nperm // 2)]
        self.radius = 1.5
        self.centers = torch.tensor(centers, dtype=torch.float32) * self.radius
        super().__init__(num_points, dim, flip_axes)
        self.bounded = False

    def create_cylinder(self, num_per_circle, std=0.1):
        nr_sphere = HyperSphere(num_per_circle, 2).data
        return torch.cat((nr_sphere, (torch.rand((num_per_circle, self.dim - 2)) - 0.5) * 2), 1)
        # nr_sphere = HyperSphere(num_per_circle, self.dim - 1).data
        # return torch.cat((nr_sphere, torch.randn(num_per_circle). view(-1, 1)), 1)

    def _create_data(self):
        num_per_circle = self.num_points // self.nperm
        self.data = torch.cat(
            [self.create_cylinder(num_per_circle) * self.radius - torch.Tensor(center)
             for center in self.centers]
        )


def plot_projection(dataset, name, shft=0):
    from matplotlib import pyplot as plt
    from nsf_utils import torchutils
    from dmatch.utils import get_top_dir

    data = torchutils.tensor2numpy(dataset.data)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bound = 4
    bounds = [[-bound, bound], [-bound, bound]]
    ax.hist2d(data[:, 0 + shft], data[:, 1 + shft], bins=256, range=bounds)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    fig.savefig(get_top_dir() + '/images/{}.png'.format(name))


def threeDscatter(dataset, nsample, name, flip=False):
    from mpl_toolkits import mplot3d
    from matplotlib import pyplot as plt
    from funnels.utils.io import get_top_dir
    dim = 3
    data = dataset(nsample, dim, flip_axes=flip).data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if flip:
        rotation_matrix = np.array([
            [0, -1],
            [1, 0]
        ], dtype=np.float32)
        data[:, :2] = data[:, :2] @ rotation_matrix
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='x', alpha=0.01)
    # fig.savefig(get_top_dir() + '/images/3D_{}.png'.format(name))
    fig.savefig(get_top_dir() + '/images/3D_{}.png'.format(name))


def plot_slices(dataset, dim, nsample, name, shft=3, width=2):
    from matplotlib import pyplot as plt
    from dmatch.utils import get_top_dir

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bound = 4
    nbins = 50
    bin_edges = np.linspace(-bound, bound, nbins + 1)
    counts = np.zeros((nbins, nbins))
    it = 0
    max_it = 100
    inds = np.arange(nsample)
    mx = np.ones(dim, dtype='bool')
    mx[[shft, shft + 1]] = 0
    while np.sum(counts) < int(1e4) and (it < max_it):
        it += 1
        data = dataset(nsample, dim).data
        # Apply a slice to the data
        to_slice = data[:, mx]
        mask = torch.all((to_slice > 0) & (to_slice < width), 1)
        data = data[mask.type(torch.bool)]
        counts += np.histogram2d(data[:, 0 + shft].numpy(), data[:, 1 + shft].numpy(), bins=bin_edges)[0]

    counts[counts == 0] = np.nan
    ax.imshow(counts.T,
              origin='lower', aspect='auto',
              extent=[-bound, bound, -bound, bound],
              )
    fig.savefig(get_top_dir() + '/images/slice_{}.png'.format(name))


def hist_features(data, name):
    import matplotlib.pyplot as plt
    from dmatch.utils import get_top_dir
    nfeatures = data.shape[1]
    fig, ax = plt.subplots(1, nfeatures, figsize=(2 + nfeatures * 5, 5))
    for i in range(nfeatures):
        ax[i].hist(data[:, i].numpy())
    print(get_top_dir() + '/images/features_hist_{}.png'.format(name))
    fig.savefig(get_top_dir() + '/images/features_hist_{}.png'.format(name))


def _test():
    threeDscatter(HyperCheckerboardDataset, int(1e5), 'unrotated', False)
    threeDscatter(HyperCheckerboardDataset, int(1e5), 'rotated', True)
    return 0


if __name__ == '__main__':
    _test()
