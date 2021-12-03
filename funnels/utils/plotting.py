# Some plotting functions
import colorsys

import numpy
import scipy
import numpy as np
import torch
from matplotlib import pyplot as plt, colors as colors

import seaborn as sns
from .torch_utils import tensor2numpy, shuffle_tensor


def get_bins(data, nbins=20):
    max_ent = data.max().item()
    min_ent = data.min().item()
    return np.linspace(min_ent, max_ent, num=nbins)


def get_mask(x, bound):
    return np.logical_and(x > bound[0], x < bound[1])

def get_weights(data):
    return np.ones_like(data) / len(data)


def apply_bound(data, bound):
    mask = np.logical_and(get_mask(data[:, 0], bound), get_mask(data[:, 1], bound))
    return data[mask, 0], data[mask, 1]


def plot2Dhist(data, ax, bins=50, bounds=None):
    if bounds:
        x, y = apply_bound(data, bounds)
    else:
        x = data[:, 0]
        y = data[:, 1]
    count, xbins, ybins = np.histogram2d(x, y, bins=bins)
    count[count == 0] = np.nan
    ax.imshow(count.T,
              origin='lower', aspect='auto',
              extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()],
              )


def getCrossFeaturePlot(data, nm, nbins=50, anomalies=None):
    nfeatures = data.shape[1]
    fig, axes = plt.subplots(nfeatures, nfeatures,
                             figsize=(np.clip(5 * nfeatures + 2, 5, 22), np.clip(5 * nfeatures - 1, 5, 20)))
    if nfeatures == 1:
        n, bins, _ = axes.hist(tensor2numpy(data), bins=nbins, alpha=0.5, density=True)
        if anomalies is not None:
            axes.hist(tensor2numpy(anomalies), bins=bins, alpha=0.5, density=True)
    else:
        for i in range(nfeatures):
            for j in range(nfeatures):
                if i == j:
                    axes[i, i].hist(tensor2numpy(data[:, i]), bins=nbins, alpha=0.5, density=True)
                    if anomalies is not None:
                        axes[i, i].hist(tensor2numpy(anomalies[:, i]), bins=nbins, alpha=0.5, density=True)
                elif i < j:
                    plot2Dhist(tensor2numpy(data[:, i:j+1]), axes[i, j], bounds=[-4, 4])
                else:
                    if anomalies is not None:
                        bini = get_bins(anomalies[:, i])
                        binj = get_bins(anomalies[:, j])
                        axes[i, j].hist2d(tensor2numpy(anomalies[:, i]), tensor2numpy(anomalies[:, j]),
                                          bins=[bini, binj], density=True, cmap='Blues')
                    else:
                        axes[i, j].set_visible(False)
    fig.tight_layout()
    plt.savefig(nm)


def plot_likelihood(array, saliency, ax, n_bins=200):
    try:
        array = array.detach().cpu().numpy()
        saliency = saliency.detach().cpu().numpy()
    except AttributeError:
        pass

    means, bins_x, bins_y, _ = scipy.stats.binned_statistic_2d(array[:, 0], array[:, 1], saliency,
                                                               'mean',
                                                               bins=n_bins
                                                               )
    # To use this as imshow you have to rotate
    means = np.rot90(means)
    extent = [array[:, 0].min(), array[:, 0].max(), array[:, 1].min(), array[:, 1].max()]
    # im = ax.imshow(means, extent=extent)
    im = ax.imshow(means, extent=extent, norm=colors.SymLogNorm(linthresh=1))
