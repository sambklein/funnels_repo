# This script, and the data it depends on, were taken from https://github.com/bayesiains/nsf/blob/master/experiments/uci.py
import argparse
import json
from contextlib import contextmanager

import numpy as np
import torch
import os

from nflows.utils import get_num_parameters, create_alternating_binary_mask
from tensorboardX import SummaryWriter
from time import sleep
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from tqdm import tqdm

import funnels.data as data_
import nflows.nn as nn_
import funnels.utils as utils

from nflows import distributions, flows, transforms

from funnels.models import sur_flows
from funnels.models.VAE import VAE
from funnels.utils.io import get_timestamp, on_cluster, get_log_root, get_checkpoint_root, save_object

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--outputdir', type=str, default='uci_local',
                    help='Choose the base output directory')
parser.add_argument('-n', '--outputname', type=str, default='local',
                    help='Set the output name directory')

# data
parser.add_argument('--dataset_name', type=str, default='gas',
                    choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'],
                    help='Name of dataset to use.')
parser.add_argument('--train_batch_size', type=int, default=512,
                    help='Size of batch used for training.')
parser.add_argument('--val_frac', type=float, default=1.,
                    help='Fraction of validation set to use.')
parser.add_argument('--val_batch_size', type=int, default=512,
                    help='Size of batch used for validation.')

# optimization
parser.add_argument('--learning_rate', type=float, default=0.0005,
                    help='Learning rate for optimizer.')
# parser.add_argument('--num_training_steps', type=int, default=200000,
#                     help='Number of total training steps.')
parser.add_argument('--num_training_steps', type=int, default=40000,
                    help='Number of total training steps.')
parser.add_argument('--anneal_learning_rate', type=int, default=1,
                    choices=[0, 1],
                    help='Whether to anneal the learning rate.')
parser.add_argument('--grad_norm_clip_value', type=float, default=5.,
                    help='Value by which to clip norm of gradients.')

# VAE details
parser.add_argument('--vae', type=int, default=0, help='Train a vae?')
parser.add_argument('--vae_width', type=int, default=512, help='VAE encoder/decoder width')
parser.add_argument('--vae_depth', type=int, default=2, help='VAE encoder/decoder depth')
parser.add_argument('--vae_drp', type=float, default=0.0, help='Dropout in VAE')
parser.add_argument('--vae_batch_norm', type=int, default=0, help='Use batch norm')
parser.add_argument('--vae_layer_norm', type=int, default=0, help='Use layer norm')

# MLP details
parser.add_argument('--mlp', type=int, default=2, help='Train a vae?')

# flow details
parser.add_argument('--base_transform_type', type=str, default='gas',
                    choices=['affine-coupling', 'quadratic-coupling', 'rq-coupling',
                             'affine-autoregressive', 'quadratic-autoregressive',
                             'rq-autoregressive'],
                    help='Type of transform to use between linear layers.')
parser.add_argument('--linear_transform_type', type=str, default='lu',
                    choices=['permutation', 'lu', 'svd'],
                    help='Type of linear transform to use.')
parser.add_argument('--num_flow_steps', type=int, default=10,
                    help='Number of blocks to use in flow.')
parser.add_argument('--hidden_features', type=int, default=256,
                    help='Number of hidden features to use in coupling/autoregressive nets.')
parser.add_argument('--tail_bound', type=float, default=3,
                    help='Box is on [-bound, bound]^2')
parser.add_argument('--num_bins', type=int, default=8,
                    help='Number of bins to use for piecewise transforms.')
parser.add_argument('--num_transform_blocks', type=int, default=2,
                    help='Number of blocks to use in coupling/autoregressive nets.')
parser.add_argument('--use_batch_norm', type=int, default=0,
                    choices=[0, 1],
                    help='Whether to use batch norm in coupling/autoregressive nets.')
parser.add_argument('--dropout_probability', type=float, default=0.25,
                    help='Dropout probability for coupling/autoregressive nets.')
parser.add_argument('--apply_unconditional_transform', type=int, default=1,
                    choices=[0, 1],
                    help='Whether to unconditionally transform \'identity\' '
                         'features in coupling layer.')
parser.add_argument('--funnel', type=int, default=4,
                    help='Whether to add a single flow layer or not.')
parser.add_argument('--funnel_level', type=int, default=2,
                    help='Whether to add a single flow layer or not.')
parser.add_argument('--cond_gauss', type=int, default=1,
                    help='Whether to add a single flow layer or not.')

# logging and checkpoints
parser.add_argument('--monitor_interval', type=int, default=500,
                    help='Interval in steps at which to report training stats.')

# reproducibility
parser.add_argument('--seed', type=int, default=1638128,
                    help='Random seed for PyTorch and NumPy.')

args = parser.parse_args()
svo = save_object(args.outputdir, args.outputname, args=args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# assert torch.cuda.is_available()
# device = torch.device('cuda')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

# create data
train_dataset = data_.load_dataset(args.dataset_name, split='train')
train_loader = data.DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    shuffle=True,
    drop_last=True
)
train_generator = data_.batch_generator(train_loader)
test_batch = next(iter(train_loader)).to(device)

# validation set
val_dataset = data_.load_dataset(args.dataset_name, split='val', frac=args.val_frac)
val_loader = data.DataLoader(
    dataset=val_dataset,
    batch_size=args.val_batch_size,
    shuffle=True,
    drop_last=True
)

# test set
test_dataset = data_.load_dataset(args.dataset_name, split='test')
test_loader = data.DataLoader(
    dataset=test_dataset,
    batch_size=args.val_batch_size,
    shuffle=False,
    drop_last=False
)

# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')

features = train_dataset.dim

is_auto_r = args.base_transform_type.split('-')[-1] == 'autoregressive'


def create_linear_transform(features):
    if args.linear_transform_type == 'permutation':
        return transforms.RandomPermutation(features=features)
    elif args.linear_transform_type == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.LULinear(features, identity_init=True)
        ])
    elif args.linear_transform_type == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.SVDLinear(features, num_householder=10, identity_init=True)
        ])
    else:
        raise ValueError


# def make_generator(dropped_entries_shape, context_shape):
#     transform_kwargs = {'tail_bound': 4., 'nstack': 3, 'nodes': 64, 'spline': True, 'tails': 'linear'}
#     return sur_flows.make_generator(dropped_entries_shape, context_shape, transform_kwargs=transform_kwargs)

def make_generator(dropped_entries_shape, context_shape):
    # transform_kwargs = {}
    base_transform_type = args.base_transform_type
    if (base_transform_type in ['rq-coupling', 'quadratic-coupling']) and (dropped_entries_shape[0] == 1):
        # If this is a 1D Flow it needs to be made autoregressive, it is the same thing in the end
        base_transform_type = base_transform_type[:-8] + 'autoregressive'
    nh_fact = 4 if is_auto_r else 3
    # nh_fact = 3
    nf_fact = 2  # if is_auto_r else 1
    n_hiddens = int(args.hidden_features / nh_fact)
    transform_kwargs = {
        'funnel': False,
        'base_transform_type': base_transform_type,
        'hidden_features': n_hiddens,
        'num_transform_blocks': int(args.num_transform_blocks),
        'num_flow_steps': int(args.num_flow_steps / nf_fact)
    }
    if args.cond_gauss:
        return sur_flows.ConditionalGaussianDecoder(dropped_entries_shape, context_shape)
        # return sur_flows.ConditionalFixedDecoder(dropped_entries_shape, context_shape)
    else:
        return sur_flows.make_generator(dropped_entries_shape, context_shape,
                                        transform_func=create_transform, transform_kwargs=transform_kwargs)


def create_base_transform(i, features, funnel=1, context_features=None, base_transform_type='rq-coupling',
                          hidden_features=128, num_transform_blocks=3):
    if funnel > 0:
        funnel_kwargs = {
            'dropped_entries_shape': [funnel],
            'context_shape': [features - funnel],
            'generator_function': make_generator,
        }
        funnel_base_model = sur_flows.BaseCouplingFunnel

        coupling_mask = torch.ones(features).byte()
        coupling_mask[-funnel:] = 0

    elif funnel == -1:
        funnel_kwargs = {
            'dropped_entries_shape': [features - 2],
            'context_shape': [2],
            'generator_function': make_generator,
        }
        funnel_base_model = sur_flows.BaseCouplingFunnelAlt

        coupling_mask = torch.zeros(features).byte()
        coupling_mask[:2] = 1

    else:
        coupling_mask = create_alternating_binary_mask(features, even=(i % 2 == 0))

    if base_transform_type == 'affine-coupling':
        model = transforms.AffineCouplingTransform(mask=coupling_mask,
                                                   transform_net_create_fn=lambda in_features,
                                                                                  out_features: nn_.nets.ResidualNet(
                                                       in_features=in_features, out_features=out_features,
                                                       hidden_features=hidden_features,
                                                       context_features=context_features,
                                                       num_blocks=num_transform_blocks, activation=F.relu,
                                                       dropout_probability=args.dropout_probability,
                                                       use_batch_norm=args.use_batch_norm)
                                                   )
        if funnel:
            model = funnel_base_model(model, **funnel_kwargs)
        return model
    elif base_transform_type == 'quadratic-coupling':
        model = transforms.PiecewiseQuadraticCouplingTransform(
            mask=coupling_mask,
            transform_net_create_fn=lambda in_features, out_features: nn_.nets.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_transform_blocks,
                activation=F.relu,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            ),
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            apply_unconditional_transform=args.apply_unconditional_transform
        )
        if funnel:
            model = funnel_base_model(model, **funnel_kwargs)
        return model
    elif base_transform_type == 'rq-coupling':
        model = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=coupling_mask,
            transform_net_create_fn=lambda in_features, out_features: nn_.nets.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=num_transform_blocks,
                activation=F.relu,
                dropout_probability=args.dropout_probability,
                use_batch_norm=args.use_batch_norm
            ),
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            apply_unconditional_transform=args.apply_unconditional_transform
        )
        if funnel:
            model = funnel_base_model(model, **funnel_kwargs)
        return model

    elif base_transform_type == 'affine-autoregressive':
        ag_model = transforms.MaskedAffineAutoregressiveTransform
        ag_params = {
            'features': features,
            'hidden_features': hidden_features,
            'context_features': context_features,
            'num_blocks': num_transform_blocks,
            'use_residual_blocks': True,
            'random_mask': False,
            'activation': F.relu,
            'dropout_probability': args.dropout_probability,
            'use_batch_norm': args.use_batch_norm
        }

    elif base_transform_type in ['quadratic-autoregressive', 'rq-autoregressive']:
        ag_model = {'quadratic-autoregressive': transforms.MaskedPiecewiseQuadraticAutoregressiveTransform,
                    'rq-autoregressive': transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform}[
            base_transform_type]
        ag_params = {
            'features': features,
            'hidden_features': hidden_features,
            'context_features': context_features,
            'num_bins': args.num_bins,
            'tails': 'linear',
            'tail_bound': args.tail_bound,
            'num_blocks': num_transform_blocks,
            'use_residual_blocks': True,
            'random_mask': False,
            'activation': F.relu,
            'dropout_probability': args.dropout_probability,
            'use_batch_norm': args.use_batch_norm
        }
    else:
        raise ValueError

    if funnel:
        # generator_kwargs = {'transform_func': }
        model = sur_flows.BaseAutoregressiveFunnel(ag_model, ag_params, args.funnel, make_generator)
    else:
        model = ag_model(**ag_params)
    return model


def create_transform(inp_dim, context_features=None, funnel=False, base_transform_type='rq-coupling',
                     hidden_features=128, num_transform_blocks=3, num_flow_steps=10):
    transform_list = []
    dim = inp_dim
    if funnel:
        # hidden_features = int(5 * hidden_features / 6)
        hidden_features = int(hidden_features)
    for i in range(num_flow_steps):
        funnel_i = funnel if (args.funnel_level == i) else 0
        transform_list += [create_linear_transform(dim)]
        transform_list += [
            create_base_transform(i, dim,
                                  funnel=funnel_i,
                                  context_features=context_features,
                                  base_transform_type=base_transform_type,
                                  hidden_features=hidden_features,
                                  num_transform_blocks=num_transform_blocks)
        ]
        if funnel_i:
            if funnel > 0:
                dim -= funnel
            else:
                dim = 2
    transform_list += [create_linear_transform(int(dim))]
    transform = transforms.CompositeTransform(transform_list)
    return transform


# create model
if args.vae:
    print('Training a VAE')
    depth = args.vae_depth
    width = args.vae_width
    # layers = [512, 512, 512]
    layers = [width] * depth
    flow = VAE(features, features - args.vae, layers, dropout=args.vae_drp, batch_norm=args.vae_batch_norm,
               layer_norm=args.vae_layer_norm)
elif args.mlp:
    print('Training a F-MLP')
    ls = features - args.mlp
    def createMLP(features):
        activ = sur_flows.SPLEEN
        activ_kwargs = {'tail_bound': 4., 'tails': 'linear', 'num_bins': 10}
        transform_list = [
            sur_flows.InferenceMLP(features, features),
            activ(**activ_kwargs),
            sur_flows.InferenceMLP(features, features),
            activ(**activ_kwargs),
            sur_flows.InferenceMLP(features, ls),
            activ(**activ_kwargs),
            sur_flows.InferenceMLP(ls, ls),
            activ(**activ_kwargs),
            sur_flows.InferenceMLP(ls, ls),
            activ(**activ_kwargs),
            sur_flows.InferenceMLP(ls, ls),
            activ(**activ_kwargs),
            sur_flows.InferenceMLP(ls, ls),
        ]
        return transforms.CompositeTransform(transform_list)


    transform = createMLP(features)
    distribution = distributions.StandardNormal((ls,))
    flow = flows.Flow(transform, distribution).to(device)

else:
    print('Training a flow or funnel.')
    if args.funnel >= 0:
        distribution = distributions.StandardNormal((int(features - args.funnel),))
    elif args.funnel == -1:
        distribution = distributions.StandardNormal((2,))
    transform = create_transform(inp_dim=features, funnel=args.funnel, base_transform_type=args.base_transform_type,
                                 hidden_features=args.hidden_features, num_transform_blocks=args.num_transform_blocks,
                                 num_flow_steps=args.num_flow_steps)
    flow = flows.Flow(transform, distribution).to(device)

n_params = get_num_parameters(flow)
print('There are {} trainable parameters in this model.'.format(n_params))

# create optimizer
optimizer = optim.Adam(flow.parameters(), lr=args.learning_rate)
if args.anneal_learning_rate:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_training_steps, 0)
else:
    scheduler = None

# create summary writer and write to log directory
timestamp = get_timestamp()
if on_cluster():
    timestamp += '||{}'.format(os.environ['SLURM_JOB_ID'])
log_dir = os.path.join(get_log_root(), args.dataset_name, f'{timestamp}_{args.outputname}')
while True:
    try:
        writer = SummaryWriter(log_dir=log_dir, max_queue=20)
        break
    except FileExistsError:
        sleep(5)
filename = os.path.join(log_dir, 'config.json')
with open(filename, 'w') as file:
    json.dump(vars(args), file)

@contextmanager
def on_cpu():
    """
    This sets a context where the default tensor type is cpu, seems necessary on Yggdrasil but not Baobab, and slows
    down evaluation significantly.
    """
    # torch.set_default_tensor_type(torch.FloatTensor)
    # yield
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    0
    yield
    0

tbar = tqdm(range(args.num_training_steps))
best_val_score = torch.tensor(-1e10)
for step in tbar:
    with on_cpu():
        batch = next(train_generator).to(device)
    flow.train()
    if args.anneal_learning_rate:
        scheduler.step()
    optimizer.zero_grad()
    log_density = flow.log_prob(batch)
    loss = - torch.mean(log_density)
    if loss.isnan():
        raise Exception('Loss is Nan.')
    loss.backward()
    if args.grad_norm_clip_value is not None:
        clip_grad_norm_(flow.parameters(), args.grad_norm_clip_value)
    optimizer.step()

    writer.add_scalar(tag='loss', scalar_value=loss.item(), global_step=step)

    if (step + 1) % args.monitor_interval == 0:
        flow.eval()

        path = os.path.join(get_checkpoint_root(args.outputdir),
                            '{}-{}'.format(args.dataset_name, args.outputname))
        torch.save(flow.state_dict(), f'{path}_model_last.pt')
        torch.save(optimizer.state_dict(), f'{path}_optimizer_last.pt')
        with open(os.path.join(get_checkpoint_root(args.outputdir), f'{args.outputname}_last_info.npy'), 'wb') as f:
            np.save(f, step)

        with torch.no_grad():
            # compute validation score
            running_val_log_density = 0
            with on_cpu():
                for val_batch in val_loader:
                    log_density_val = flow.log_prob(val_batch.to(device).detach())
                    mean_log_density_val = torch.mean(log_density_val).detach()
                    running_val_log_density += mean_log_density_val
            running_val_log_density /= len(val_loader)

        if running_val_log_density > best_val_score:
            best_val_score = running_val_log_density
            path = os.path.join(get_checkpoint_root(args.outputdir),
                                '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
            torch.save(flow.state_dict(), path)

        # compute reconstruction
        with torch.no_grad():
            test_batch_noise = flow.transform_to_noise(test_batch)
            test_batch_reconstructed, _ = flow._transform.inverse(test_batch_noise)
        errors = test_batch - test_batch_reconstructed
        max_abs_relative_error = torch.abs(errors / test_batch).max()
        average_abs_relative_error = torch.abs(errors / test_batch).mean()
        writer.add_scalar('max-abs-relative-error',
                          max_abs_relative_error, global_step=step)
        writer.add_scalar('average-abs-relative-error',
                          average_abs_relative_error, global_step=step)

        summaries = {
            'val': running_val_log_density.item(),
            'best-val': best_val_score.item(),
            'max-abs-relative-error': max_abs_relative_error.item(),
            'average-abs-relative-error': average_abs_relative_error.item()
        }
        for summary, value in summaries.items():
            writer.add_scalar(tag=summary, scalar_value=value, global_step=step)

# load best val model
path = os.path.join(get_checkpoint_root(args.outputdir),
                    '{}-best-val-{}.t'.format(args.dataset_name, timestamp))
flow.load_state_dict(torch.load(path))
flow.eval()

# calculate log-likelihood on test set
with torch.no_grad():
    log_likelihood = torch.Tensor([])
    for batch in tqdm(test_loader):
        with set_default_tensor_type():
            log_density = flow.log_prob(batch.to(device)) 
            log_likelihood = torch.cat([
                log_likelihood,
                log_density
            ])
path = os.path.join(log_dir, '{}-{}-log-likelihood.npy'.format(
    args.dataset_name,
    args.base_transform_type
))
np.save(path, utils.tensor2numpy(log_likelihood))
mean_log_likelihood = log_likelihood.mean()
std_log_likelihood = log_likelihood.std()

# save log-likelihood
s = 'Final score for {}: {:.2f} +- {:.2f}'.format(
    args.dataset_name.capitalize(),
    mean_log_likelihood.item(),
    2 * std_log_likelihood.item() / np.sqrt(len(test_dataset))
)
print(s)
filename = os.path.join(log_dir, 'test-results.txt')
with open(filename, 'w') as file:
    file.write(s)
