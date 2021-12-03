from nflows import transforms
from torch.nn import functional as F

from funnels.models.nn.MLPs import dense_net


def get_transform(inp_dim=1, nodes=64, num_blocks=2, nstack=2, tails='linear', tail_bound=1., num_bins=10,
                  context_features=1, lu=1, bnorm=1, spline=True, activation=F.leaky_relu):
    transform_list = []
    for i in range(nstack):

        if tails is not None:
            tb = tail_bound
        else:
            tb = tail_bound if i == 0 else None

        if spline:
            transform_list += [
                transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, nodes,
                                                                                   num_blocks=num_blocks,
                                                                                   tail_bound=tb, num_bins=num_bins,
                                                                                   tails=tails,
                                                                                   context_features=context_features,
                                                                                   activation=activation)]
        else:
            transform_list += [transforms.MaskedAffineAutoregressiveTransform(inp_dim, nodes, num_blocks=num_blocks,
                                                                              activation=activation,
                                                                              context_features=context_features)]

        # if bnorm:
        #     transform_list += [transforms.BatchNorm(inp_dim)]

        if (tails is None) and (tail_bound is not None) and (i == nstack - 1):
            transform_list += [transforms.standard.PointwiseAffineTransform(-tail_bound, 2 * tail_bound)]

        if lu:
            transform_list += [transforms.LULinear(inp_dim)]
        else:
            transform_list += [transforms.ReversePermutation(inp_dim)]

    return transforms.CompositeTransform(transform_list[:-1])


def maker_net(input_dim, output_dim):
    return dense_net(input_dim, output_dim, layers=[128] * 3)


def coupling_spline(inp_dim, maker=maker_net, nstack=3, tail_bound=None, tails=None, activation=F.relu, lu=0,
                    num_bins=10, mask=[1, 0], unconditional_transform=True):
    transform_list = []
    for i in range(nstack):
        # If a tail function is passed apply the same tail bound to every layer, if not then only use the tail bound on
        # the final layer
        tpass = tails
        if tails:
            tb = tail_bound
        else:
            tb = tail_bound if i == 0 else None
        transform_list += [
            transforms.PiecewiseRationalQuadraticCouplingTransform(mask, maker, tail_bound=tb, num_bins=num_bins,
                                                                   tails=tpass,
                                                                   apply_unconditional_transform=unconditional_transform)]
        if (tails == None) and (not tail_bound == None) and (i == nstack - 1):
            transform_list += [transforms.standard.PointwiseAffineTransform(-tail_bound, 2 * tail_bound)]

        if lu:
            transform_list += [transforms.LULinear(inp_dim)]
        else:
            transform_list += [transforms.ReversePermutation(inp_dim)]

    return transforms.CompositeTransform(transform_list[:-1])


import nflows.nn as nn_
from nflows.utils import get_num_parameters, create_alternating_binary_mask


def get_transform_full(inp_dim=1, nodes=64, num_blocks=2, nstack=2, tails=None, tail_bound=1., num_bins=10,
                       context_features=1, lu=1, bnorm=1, model='rq_coupling', activation=F.leaky_relu,
                       dropout_probability=0.0):
    transform_list = []
    for i in range(nstack):

        if tails is not None:
            tb = tail_bound
        else:
            tb = tail_bound if i == 0 else None

        if model == 'affine-coupling':
            transform_list += [
                transforms.AffineCouplingTransform(mask=create_alternating_binary_mask(inp_dim, even=(i % 2 == 0)),
                                                   transform_net_create_fn=lambda in_features,
                                                                                  out_features: nn_.nets.ResidualNet(
                                                       in_features=in_features,
                                                       out_features=out_features,
                                                       hidden_features=nodes,
                                                       context_features=context_features,
                                                       num_blocks=num_blocks,
                                                       activation=F.relu,
                                                       dropout_probability=dropout_probability,
                                                       use_batch_norm=bnorm)
                                                   )
            ]
        else:
            transform_list += [transforms.MaskedAffineAutoregressiveTransform(inp_dim, nodes, num_blocks=num_blocks,
                                                                              activation=activation,
                                                                              context_features=context_features)]

        # if bnorm:
        #     transform_list += [transforms.BatchNorm(inp_dim)]

        if (tails is None) and (tail_bound is not None) and (i == nstack - 1):
            transform_list += [transforms.standard.PointwiseAffineTransform(-tail_bound, 2 * tail_bound)]

        if lu:
            transform_list += [transforms.LULinear(inp_dim)]
        else:
            transform_list += [transforms.ReversePermutation(inp_dim)]

    return transforms.CompositeTransform(transform_list[:-1])
