import torch
from nflows import distributions, flows, transforms
from nflows.utils import create_mid_split_binary_mask
import torch.nn as nn
from nflows.nn.nets import ConvResidualNet


class Conv2dSameSize(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size):
        same_padding = kernel_size // 2  # Padding that would keep the spatial dims the same
        super().__init__(in_channels, out_channels, kernel_size,
                         padding=same_padding)


class ConvNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            Conv2dSameSize(in_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            Conv2dSameSize(hidden_channels, out_channels, kernel_size=3),
        )

    def forward(self, inputs, context=None):
        return self.net.forward(inputs)


spline_params = {
    'num_bins': 5,
    'tail_bound': 1.,
    'min_bin_width': 1e-3,
    'min_bin_height': 1e-3,
    'min_derivative': 1e-3,
    'apply_unconditional_transform': False
}


def create_transform_step(num_channels, hidden_channels, coupling_layer_type='rational_quadratic_spline',
                          context_channels=None, actnorm=True, use_resnet=True, dropout_prob=0.,
                          num_res_blocks=3, resnet_batchnorm=True, spline_params=spline_params):
    if use_resnet:
        def create_convnet(in_channels, out_channels):
            net = ConvResidualNet(in_channels=in_channels,
                                  out_channels=out_channels,
                                  hidden_channels=hidden_channels,
                                  num_blocks=num_res_blocks,
                                  use_batch_norm=resnet_batchnorm,
                                  dropout_probability=dropout_prob,
                                  context_channels=context_channels)
            return net
    else:
        if dropout_prob != 0.:
            raise ValueError()

        def create_convnet(in_channels, out_channels):
            return ConvNet(in_channels, hidden_channels, out_channels)

    mask = create_mid_split_binary_mask(num_channels)

    if coupling_layer_type == 'cubic_spline':
        coupling_layer = transforms.PiecewiseCubicCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height']
        )
    elif coupling_layer_type == 'quadratic_spline':
        coupling_layer = transforms.PiecewiseQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height']
        )
    elif coupling_layer_type == 'rational_quadratic_spline':
        coupling_layer = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height'],
            min_derivative=spline_params['min_derivative']
        )
    elif coupling_layer_type == 'affine':
        coupling_layer = transforms.AffineCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet
        )
    elif coupling_layer_type == 'additive':
        coupling_layer = transforms.AdditiveCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet
        )
    else:
        raise RuntimeError('Unknown coupling_layer_type')

    step_transforms = []

    if actnorm:
        step_transforms.append(transforms.ActNorm(num_channels))

    step_transforms.extend([
        transforms.OneByOneConvolution(num_channels),
        coupling_layer
    ])

    return transforms.CompositeTransform(step_transforms)


class ReshapeTransform(transforms.Transform):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs, context=None):
        if tuple(inputs.shape[1:]) != self.input_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.input_shape))
        return inputs.reshape(-1, *self.output_shape), torch.zeros(inputs.shape[0]).to(inputs.device)

    def inverse(self, inputs, context=None):
        if tuple(inputs.shape[1:]) != self.output_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.output_shape))
        return inputs.reshape(-1, *self.input_shape), torch.zeros(inputs.shape[0]).to(inputs.device)


def add_glow(size_in, levels, hidden_channels, context_channels=None, steps_per_level=3):
    c, h, w = size_in
    all_transforms = []
    for level, level_hidden_channels in zip(range(levels), hidden_channels):
        image_size = c * h * w
        squeeze_transform = transforms.SqueezeTransform()
        # c_t, h_t, w_t = squeeze_transform.get_output_shape(c, h, w)
        # if c_t * h_t * w_t == image_size:
        #     squeeze = 1
        #     c, h, w = c_t, h_t, w_t
        # else:
        #     print(f'No more squeezing after level {level}')
        #     squeeze = 0
        squeeze = 0

        layer_transform = [create_transform_step(c, level_hidden_channels, context_channels=context_channels) for _ in
                           range(steps_per_level)] + [transforms.OneByOneConvolution(c)]
        if squeeze:
            layer_transform = [squeeze_transform] + layer_transform
        all_transforms += [transforms.CompositeTransform(layer_transform)]

    all_transforms.append(ReshapeTransform(
        input_shape=(c, h, w),
        output_shape=(c * h * w,)
    ))

    return all_transforms


def create_transform(size_in, context_channels=None, num_bits=8, preprocessing='glow', levels=3, hidden_channels=64):
    # if not isinstance(hidden_channels, list):
    #     hidden_channels = [hidden_channels] * levels

    c, h, w = size_in

    hc = [hidden_channels] * levels
    mct = transforms.CompositeTransform(add_glow(size_in, levels, hc, context_channels=context_channels))

    # Inputs to the model in [0, 2 ** num_bits]

    if preprocessing == 'glow':
        # Map to [-0.5,0.5]
        preprocess_transform = transforms.AffineScalarTransform(scale=1. / 2 ** num_bits,
                                                                shift=-0.5)
    else:
        raise RuntimeError('Unknown preprocessing type: {}'.format(preprocessing))

    return transforms.CompositeTransform([preprocess_transform, mct]), (c, h, w)


def create_flow(size_in, context_channels=None, flow_checkpoint=None):
    if isinstance(context_channels, list) or isinstance(context_channels, tuple):
        context_channels = context_channels[0]
    transform, (c_out, h_out, w_out) = create_transform(size_in, context_channels=context_channels)
    distribution = distributions.StandardNormal((c_out * h_out * w_out,))

    flow = flows.Flow(transform, distribution)

    if flow_checkpoint is not None:
        flow.load_state_dict(torch.load(flow_checkpoint))

    return flow
