import math
import torch
from funnels.utils.torch_utils import tensor2numpy


def nats_to_bits_per_dim(nats, c, h, w):
    return nats / (math.log(2) * c * h * w)

def eval_log_density(log_prob_fn, data_loader, num_batches=None):
    with torch.no_grad():
        total_ld = 0
        batch_counter = 0
        for batch in data_loader:
            if isinstance(batch, list): # If labelled dataset, ignore labels
                batch = batch[0]
            log_prob = log_prob_fn(batch)
            total_ld += torch.mean(log_prob)
            batch_counter += 1
            if (num_batches is not None) and batch_counter == num_batches:
                break
        return total_ld / batch_counter


def progress_string(elapsed_time, step, num_steps):
    rate = step / elapsed_time
    if rate > 0:
        remaining_time = format_interval((num_steps - step) / rate)
    else:
        remaining_time = '...'
    elapsed_time = format_interval(elapsed_time)
    return '{}<{}, {:.2f}it/s'.format(elapsed_time, remaining_time, rate)


# From https://github.com/tqdm/tqdm/blob/master/tqdm/_tqdm.py
def format_interval(t):
    """
    Formats a number of seconds as a clock time, [H:]MM:SS
    Parameters
    ----------
    t  : int
        Number of seconds.
    Returns
    -------
    out  : str
        [H:]MM:SS
    """
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return '{0:d}:{1:02d}:{2:02d}'.format(h, m, s)
    else:
        return '{0:02d}:{1:02d}'.format(m, s)


def imshow(image, ax):
    image = tensor2numpy(image.permute(1, 2, 0))

    if image.shape[-1] == 1:
        ax.imshow(1 - image[..., 0], cmap='Greys')
    else:
        ax.imshow(image)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
