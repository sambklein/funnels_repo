from nflows import transforms
from nflows import flows
import nflows
import torch
import torch.nn as nn
from nflows.transforms import splines, PiecewiseRationalQuadraticCDF
from nflows.transforms.coupling import CouplingTransform
from nflows.utils import sum_except_batch

from funnels.models.GLOW import create_flow
from funnels.models.flows import get_transform
from funnels.models.nn.MLPs import dense_net
import numpy as np

import torch


class StochasticPermutation(nflows.transforms.Transform):
    '''A stochastic permutation layer.'''

    def __init__(self, dim=1):
        super(StochasticPermutation, self).__init__()
        self.register_buffer('buffer', torch.zeros(1))
        self.dim = dim

    def forward(self, x):
        rand = torch.rand(x.shape[0], x.shape[self.dim], device=x.device)
        permutation = rand.argsort(dim=1)
        for d in range(1, self.dim):
            permutation = permutation.unsqueeze(1)
        for d in range(self.dim + 1, x.dim()):
            permutation = permutation.unsqueeze(-1)
        permutation = permutation.expand_as(x)
        z = torch.gather(x, self.dim, permutation)
        ldj = self.buffer.new_zeros(x.shape[0])
        return z, ldj

    def inverse(self, z):
        rand = torch.rand(z.shape[0], z.shape[self.dim], device=z.device)
        permutation = rand.argsort(dim=1)
        for d in range(1, self.dim):
            permutation = permutation.unsqueeze(1)
        for d in range(self.dim + 1, z.dim()):
            permutation = permutation.unsqueeze(-1)
        permutation = permutation.expand_as(z)
        x = torch.gather(z, self.dim, permutation)
        ldj = self.buffer.new_zeros(x.shape[0])
        return x, ldj


def not_an_image_exception():
    raise Exception('Convolutions only work on images, need a channel dimension even if there is only one channel.')


def convolving_not_possible(size):
    raise RuntimeError(f'Cannot properly reshape images of {size} size for this convolution.')


class get_net(nn.Module):

    def __init__(self, features, hidden_features, num_blocks, output_multiplier):
        super(get_net, self).__init__()
        self.feature_list = [1, 1]
        self.makers = nn.ModuleList(
            [dense_net(self.feature_list[i], output_multiplier, layers=[hidden_features] * num_blocks) for i in
             range(features)])

    def forward(self, data, context=None):
        splines = []
        for i, function in enumerate(self.makers):
            # All outputs are a function of the dimension which is dropped, garuantees right invertibility.
            splines += [function(data[:, 0].view(-1, 1))]
        return torch.cat(splines, 1)


class SurNSF(nflows.transforms.Transform):

    def __init__(self, features, hidden_features, num_blocks=2, num_bins=10, tail_bound=4., tails='linear', spline=True,
                 **kwargs):
        super(SurNSF, self).__init__()

        self.features = features
        inp_dim = 1
        nstack = 2
        self.flow_transform = get_transform(inp_dim=inp_dim,
                                            nodes=hidden_features,
                                            num_blocks=num_blocks,
                                            tails=tails,
                                            num_bins=num_bins,
                                            tail_bound=tail_bound,
                                            nstack=nstack,
                                            context_features=features - 1,
                                            spline=spline)
        base_dist = nflows.distributions.StandardNormal([inp_dim])
        self.one_dim_flow = flows.Flow(self.flow_transform, base_dist)

        self.transform = get_transform(inp_dim=features - 1, context_features=1, tails='linear', spline=spline)

    def forward(self, inputs, context=None):
        input_dropped = inputs[:, 0].view(-1, 1)
        output, jacobian = self.transform.forward(inputs[:, 1:].view(-1, self.features - 1),
                                                  context=input_dropped)
        likelihood_contribution = self.one_dim_flow.log_prob(input_dropped, context=output)
        return output, jacobian + likelihood_contribution

    def inverse(self, inputs, context=None):
        input_dropped = self.one_dim_flow.sample(1, context=inputs).squeeze().view(-1, 1)
        input_mapped, jacobian = self.transform.inverse(inputs, context=input_dropped)
        likelihood_contribution = self.one_dim_flow.log_prob(input_dropped.view(-1, 1), context=inputs)
        f_return = torch.cat((input_dropped, input_mapped), 1)
        return f_return, jacobian + likelihood_contribution


class IdentityTransform(nflows.transforms.Transform):
    """
    An N x 1 funnel convolution with the stride fixed to N.
    """

    def forward(self, inputs, context=None):
        return inputs, inputs.new_zeros(inputs.shape[0]).to(inputs.device)

    def inverse(self, inputs, context=None):
        return inputs, inputs.new_zeros(inputs.shape[0]).to(inputs.device)


class FlattenTransform(nflows.transforms.Transform):

    def __init__(self, c, h, w, *args, **kwargs):
        super(FlattenTransform, self).__init__(*args, **kwargs)
        self.c = c
        self.h = h
        self.w = w

    def forward(self, inputs, context=None):
        batch_size = inputs.shape[0]
        return inputs.view(batch_size, -1), inputs.new_zeros(batch_size).to(inputs.device)

    def inverse(self, inputs, context=None):
        batch_size = inputs.shape[0]
        return inputs.view(batch_size, self.c, self.h, self.w), inputs.new_zeros(batch_size).to(inputs.device)


def product_except_batch(x):
    """Take product of all dimensions except the first."""
    batch_size = x.shape[0]
    return x.view(batch_size, -1).prod(1)


class SurFlow(flows.Flow):
    def __init__(self, transform, distribution, embedding_net=None, decoder=None):
        super(SurFlow, self).__init__(transform, distribution, embedding_net=embedding_net)
        self.decoder = decoder

    def _sample(self, num_samples, context):
        if self.decoder is None:
            return super(SurFlow, self)._sample(num_samples, context)
        else:
            base_samples = self._distribution.sample(num_samples)
            return self.decoder.sample(1, context=base_samples).squeeze()

    def _log_prob(self, inputs, context):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        log_prob = self._distribution.log_prob(noise, context=embedded_context)
        total_log_prob = log_prob + logabsdet
        return total_log_prob


class InferenceMLP(nflows.transforms.Transform):
    def __init__(self, in_nodes, out_nodes, width=128, depth=3):
        super(InferenceMLP, self).__init__()
        self.perm = transforms.RandomPermutation(features=in_nodes)
        self.F = transforms.LULinear(out_nodes)
        self.dim_reduc = in_nodes > out_nodes
        if self.dim_reduc:
            self.V = nn.Linear(in_nodes - out_nodes, out_nodes)
            self.decoder = ConditionalGaussianDecoder(in_nodes - out_nodes, out_nodes, width=width, depth=depth)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

    def forward(self, x, context=None):
        x, l_rand = self.perm(x, context=context)
        xPlus = x[:, :self.out_nodes]
        output, likelihood_contr = self.F(xPlus, context=context)
        x_minus = x[:, self.out_nodes:]
        if self.dim_reduc:
            cond_part = self.V(x_minus)
            output = output + cond_part
            likelihood_cond = self.decoder.log_prob(x_minus, context=output)
        else:
            likelihood_cond = 0
        return output, likelihood_contr + likelihood_cond + l_rand

    def inverse(self, z, context=None):
        if self.dim_reduc:
            samples, log_prob = self.decoder.sample_and_log_prob(1, context=z)
            cond_part = self.V(samples.squeeze())
        else:
            cond_part = 0
            log_prob = z.new_zeros(z.shape[0])
        x, like = self.F.inverse(z - cond_part, context=context)
        if self.dim_reduc:
            x = torch.cat((x, samples.squeeze()), 1)
        x, l_rand = self.perm.inverse(x, context=context)
        return x, log_prob.view(-1) + like + l_rand


class TanhLayer(nflows.transforms.Transform):
    """
    Tanh activation as a survae layer, this assumes each point is independent and so the total log contribution
    factorises, leaving the final contribution to just be the sum of each individual contribution.
    """

    def forward(self, x, context=None):
        z = torch.tanh(x)
        detJ = 1 - torch.tanh(x) ** 2
        return z, sum_except_batch(detJ.abs().log())

    def inverse(self, z, context=None):
        x = torch.atanh(z)
        detJ = -torch.tanh(x) ** 2 * (1 - torch.tanh(x) ** 2)
        return x, sum_except_batch(detJ.abs().log())


class LeakyRelu(nflows.transforms.Transform):
    """
    LeakyRelu activation as a survae layer, this assumes each point is independent and so the total log contribution
    factorises, leaving the final contribution to just be the sum of each individual contribution.
    """

    def __init__(self, learnable=False):
        super(LeakyRelu, self).__init__()
        if learnable:
            self.neg_slope = nn.Parameter(torch.randn(1) ** (1 / 2))
        else:
            self.register_buffer('neg_slope', torch.tensor((0.01) ** (1 / 2), dtype=torch.float32))
        self.register_buffer('epsilon', torch.tensor(1e-6, dtype=torch.float32))

    def get_func(self, x):
        leaky_relu = torch.ones_like(x)
        leaky_relu[x < 0] = (self.neg_slope + self.epsilon) ** 2
        return leaky_relu

    def forward(self, x, context=None):
        leaky_relu = self.get_func(x)
        z = x * leaky_relu
        detJ = leaky_relu
        return z, sum_except_batch(detJ.abs().log())

    def inverse(self, z, context=None):
        leaky_relu = self.get_func(z)
        x = z / leaky_relu
        detJ = 1 / leaky_relu
        return x, sum_except_batch(detJ.abs().log())


class SPLEEN(PiecewiseRationalQuadraticCDF):

    def __init__(self, tail_bound=1., tails=None, num_bins=10):
        super(SPLEEN, self).__init__(1, tail_bound=tail_bound, tails=tails, num_bins=num_bins)

    def forward(self, inputs, context=None):
        sh = inputs.shape
        out, contr = super(SPLEEN, self).forward(inputs.view(-1, 1))
        return out.view(sh), contr.view(sh).sum(-1)

    def inverse(self, inputs, context=None):
        sh = inputs.shape
        out, contr = super(SPLEEN, self).inverse(inputs.view(-1, 1))
        return out.view(sh), contr.view(sh).sum(-1)


class NByOneStandardConv(nflows.transforms.Transform):
    """
    A width x width funnel convolution with the stride fixed to width.
    """

    def __init__(self, num_channels, image_width, width=2, hidden_features=128, num_blocks=2,
                 num_bins=10, tail_bound=1., tails='linear', nstack=10, spline=1, transform=None, gauss=True,
                 **kwargs):
        super(NByOneStandardConv, self).__init__()

        self.padding = 0
        self.stride = width
        self.forward_convolution = nn.Conv2d(num_channels, num_channels, width, stride=self.stride,
                                             padding=self.padding)
        self.unfold = nn.Unfold(kernel_size=width, stride=self.stride, padding=self.padding)
        self.fold = nn.Fold(output_size=(image_width, image_width), kernel_size=width, stride=self.stride,
                            padding=self.padding)
        # k_infer = 5
        # self.gather_z = nn.Unfold(kernel_size=k_infer, stride=1, padding=2)\
        k_infer = 3
        self.gather_z = nn.Unfold(kernel_size=3, stride=1, padding=1)
        self.n_cond = num_channels * k_infer ** 2
        # self.n_cond = num_channels
        self.num_channels = num_channels
        self.width = width
        self.n_dropped = num_channels * (width ** 2 - 1)
        self.output_image_size = int((image_width + self.padding) / 2)

        mx = torch.ones(self.width ** 2 * self.num_channels + 1, dtype=torch.bool)
        mx[::self.width ** 2] = 0
        self.mx = mx[1:]

        if gauss:
            self.decoder = ConditionalGaussianDecoder(num_channels * (width ** 2 - 1), self.n_cond)
            # self.decoder = ConditionalFixedDecoder(num_channels * (width ** 2 - 1), self.n_cond)
        else:
            self.decoder = self.get_one_dim_flow(num_channels * (width ** 2 - 1), self.n_cond, width, nstack,
                                                 hidden_features, num_blocks, tail_bound,
                                                 num_bins, tails, spline=spline)

    def get_one_dim_flow(self, inp_dim, ncond, width, nstack, hidden_features, num_blocks, tail_bound, num_bins, tails,
                         spline=True):
        transform_list = []
        for i in range(nstack):
            if spline:
                transform_list += [
                    transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(inp_dim, hidden_features,
                                                                                       num_blocks=num_blocks,
                                                                                       tail_bound=tail_bound,
                                                                                       num_bins=num_bins,
                                                                                       tails=tails,
                                                                                       context_features=ncond)]
            else:
                transform_list += [transforms.MaskedAffineAutoregressiveTransform(inp_dim, 128, num_blocks=4,
                                                                                  context_features=width - 1)]

            transform_list += [transforms.ReversePermutation(inp_dim)]

        flow_transform = transforms.CompositeTransform(transform_list[:-1])
        base_dist = nflows.distributions.StandardNormal([inp_dim])
        return flows.Flow(flow_transform, base_dist)

    def get_J(self):
        return self.forward_convolution.weight[..., -1, -1]

    def forward(self, inputs, context=None):
        if inputs.dim() != 4:
            not_an_image_exception()
        batch_size, c, h, w = inputs.shape
        z = self.forward_convolution(inputs)
        log_detJ = self.get_J().det().abs().log() * np.prod(z.shape[-2:])
        transformed_blocks = self.unfold(inputs).transpose(-2, -1)
        dropped_sections = transformed_blocks[..., self.mx]
        tiles = self.gather_z(z).transpose(1, 2).reshape(-1, self.n_cond)
        cond_likelihood = self.decoder.log_prob(
            dropped_sections.view(-1, self.n_dropped),
            context=tiles
        )
        cond_likelihood = cond_likelihood.view(batch_size, -1).sum(-1)
        likelihood_contribution = log_detJ + cond_likelihood
        return z, likelihood_contribution

    def make_an_image(self, batch_size, samples, inverse_s=None):
        samples = samples.view(batch_size, -1, self.n_dropped)
        x_preform = samples.new_zeros(*samples.shape[:-1], len(self.mx))
        x_preform[..., self.mx] = samples
        if inverse_s is not None:
            x_preform[..., ~self.mx] = inverse_s
        image = self.fold(x_preform.transpose(-2, -1))
        return image

    def inverse(self, z, context=None):
        if z.dim() != 4:
            not_an_image_exception()
        batch_size, c, h, w = z.shape
        samples, log_prob = self.decoder.sample_and_log_prob(
            1, context=self.gather_z(z).transpose(1, 2).reshape(-1, self.n_cond)
        )
        dropped_sections = self.make_an_image(batch_size, samples)
        consts = self.forward_convolution(dropped_sections)
        inv_J = self.get_J().inverse()
        transformed_sections = torch.einsum('mn,inkl->imkl', inv_J, z - consts)
        x = self.make_an_image(batch_size, samples,
                               transformed_sections.view(batch_size, self.num_channels,
                                                         int(self.output_image_size ** 2)).transpose(1, 2))
        log_detJ = inv_J.det().abs().log() * np.prod(z.shape[-2:])
        return x, log_prob.view(batch_size, -1).sum(-1) + log_detJ

    def test_inverse(self, inputs):
        batch_size, c, h, w = inputs.shape
        z = self.forward_convolution(inputs)
        transformed_blocks = self.unfold(inputs).transpose(-2, -1)
        samples = transformed_blocks[..., self.mx].view(-1, self.n_dropped)
        dropped_sections = self.make_an_image(batch_size, samples)
        consts = self.forward_convolution(dropped_sections)
        inv_J = self.get_J().inverse()
        transformed_sections = torch.einsum('mn,inkl->imkl', inv_J, z - consts)
        x = self.make_an_image(batch_size, samples, transformed_sections.view(batch_size, 3, 256).transpose(1, 2))
        return x

class MakeAnImage(nflows.transforms.Transform):

    def forward(self, inputs, context=None):
        batch_size, width = inputs.shape
        return inputs.view(batch_size, 1, 1, width), inputs.new_zeros(batch_size)

    def inverse(self, inputs, context=None):
        batch_size, _, _, width = inputs.shape
        return inputs.view(batch_size, width), inputs.new_zeros(batch_size)


class UnMakeAnImage(MakeAnImage):
    def forward(self, inputs, context=None):
        return super(UnMakeAnImage, self).inverse(inputs)

    def inverse(self, inputs, context=None):
        return super(UnMakeAnImage, self).forward(inputs)


transform_kwargs = {'tail_bound': 4., 'nstack': 3, 'nodes': 64, 'spline': True, 'tails': 'linear'}


class make_generator(flows.Flow):

    def __init__(self, dropped_entries_shape, context_shape, transform_func=get_transform, transform_kwargs=None):

        """
        :param dropped_entries_shape: the shape of the data that needs to be sampled and evaluated (for likelihood)
        :param context_shape: the shape of the data that will be passed as context
        :return: a flow capable of generating, and evaluating the likelihood, data of shape dropped_entries_shape given
                 data of shape context_shape as context.
        """
        if not isinstance(transform_kwargs, dict):
            transform_kwargs = {}
        self.dropped_entries_shape = self.make_list(dropped_entries_shape)
        self.context_shape = self.make_list(context_shape)
        self.input_size = int(np.prod(dropped_entries_shape))
        self.context_size = int(np.prod(context_shape))
        transform = transform_func(inp_dim=self.input_size, context_features=self.context_size, **transform_kwargs)
        base_dist = nflows.distributions.StandardNormal([self.input_size])
        # base_dist = nflows.distributions.uniform.BoxUniform(-self.tail_bound * torch.ones(self.input_size),
        #                                                     self.tail_bound * torch.ones(self.input_size))
        super(make_generator, self).__init__(transform, base_dist)

    def make_list(self, var):
        if not isinstance(var, list):
            var = [var]
        return var

    def _log_prob(self, inputs, context):
        inputs = inputs.view(-1, self.input_size)
        context = context.view(-1, self.context_size)
        return super(make_generator, self)._log_prob(inputs, context)

    def _sample(self, num_samples, context):
        context = context.view(-1, self.context_size)
        return super(make_generator, self)._sample(num_samples, context).view(-1, *self.dropped_entries_shape)


# class ConditionalGaussianDecoder(nflows.distributions.Distribution):
#
#     def __init__(self, dropped_entries_shape, context_shape, transform_kwargs=None, width=128, depth=3):
#         """
#         :param dropped_entries_shape: the shape of the data that needs to be sampled and evaluated (for likelihood)
#         :param context_shape: the shape of the data that will be passed as context
#         :return: a flow capable of generating, and evaluating the likelihood, data of shape dropped_entries_shape given
#                  data of shape context_shape as context.
#         """
#         super(ConditionalGaussianDecoder, self).__init__()
#         if not isinstance(transform_kwargs, dict):
#             transform_kwargs = {}
#         self.output_size = int(np.prod(dropped_entries_shape))
#         self.context_size = int(np.prod(context_shape))
#         self.rpi = 0.5 * torch.log(torch.tensor(2 * np.pi))
#
#         self.net = dense_net(self.context_size, self.output_size * 2, layers=[width] * depth)
#         # self.net = dense_net(self.context_size, self.output_size * 2, layers=[64] * 3)
#
#     def get_param(self, context):
#         context = context.view(-1, self.context_size)
#         return self.net(context).split(self.output_size, dim=1)
#
#     def _log_prob(self, inputs, context):
#         mean, log_sigma = self.get_param(context)
#         sigma = torch.exp(-log_sigma)
#         inputs = inputs.view(-1, self.output_size)
#         log_prob = -0.5 * ((mean - inputs) / sigma) ** 2 - log_sigma - self.rpi
#         return log_prob.sum(-1)
#
#     def _sample(self, num_samples, context):
#         # Ignore num_samples as it is always defined by the size of the context
#         mean, log_sigma = self.get_param(context)
#         sigma = torch.exp(log_sigma)
#         epsilon = torch.normal(mean=torch.zeros_like(mean), std=torch.ones_like(mean))
#         return (mean + sigma * epsilon).unsqueeze(0)
#         # return mean.unsqueeze(0)


class ConditionalGaussianDecoder(nflows.distributions.ConditionalDiagonalNormal):

    def __init__(self, dropped_entries_shape, context_shape, transform_kwargs=None, width=128, depth=3):
        """
        :param dropped_entries_shape: the shape of the data that needs to be sampled and evaluated (for likelihood)
        :param context_shape: the shape of the data that will be passed as context
        :return: a flow capable of generating, and evaluating the likelihood, data of shape dropped_entries_shape given
                 data of shape context_shape as context.
        """
        self.output_size = int(np.prod(dropped_entries_shape))
        self.context_size = int(np.prod(context_shape))
        super(ConditionalGaussianDecoder, self).__init__([dropped_entries_shape],
                                                         dense_net(self.context_size, self.output_size * 2,
                                                                   layers=[width] * depth))

    def _log_prob(self, inputs, context):
        lp = super(ConditionalGaussianDecoder, self)._log_prob(inputs, context)
        return torch.clip(lp, min=-10000)


class ConditionalFixedDecoder(nflows.distributions.Distribution):

    def __init__(self, dropped_entries_shape, context_shape, transform_kwargs=None, sigma=0.1, width=256, depth=3):
        """
        :param dropped_entries_shape: the shape of the data that needs to be sampled and evaluated (for likelihood)
        :param context_shape: the shape of the data that will be passed as context
        :return: a flow capable of generating, and evaluating the likelihood, data of shape dropped_entries_shape given
                 data of shape context_shape as context.
        """
        super(ConditionalFixedDecoder, self).__init__()
        if not isinstance(transform_kwargs, dict):
            transform_kwargs = {}
        self.output_size = int(np.prod(dropped_entries_shape))
        self.context_size = int(np.prod(context_shape))
        self.rpi = torch.tensor((2 * np.pi) ** (1 / 2), dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)

        self.net = dense_net(self.context_size, self.output_size, layers=[width] * depth)
        # self.net = dense_net(self.context_size, self.output_size, layers=[64] * 3)

    def get_param(self, context):
        context = context.view(-1, self.context_size)
        return self.net(context)

    def _log_prob(self, inputs, context):
        mean = self.get_param(context)
        sigma = self.sigma
        inputs = inputs.view(-1, self.output_size)
        log_prob = -0.5 * ((mean - inputs) / sigma) ** 2 - torch.log(sigma * self.rpi)
        return log_prob.sum(-1)

    def _sample(self, num_samples, context):
        # Ignore num_samples as it is always defined by the size of the context
        mean = self.get_param(context)
        epsilon = torch.normal(mean=torch.zeros_like(mean), std=torch.ones_like(mean))
        return (mean + self.sigma * epsilon).unsqueeze(0)
        # return mean.unsqueeze(0)

class SurVaeCoupling(CouplingTransform):

    def __init__(self, dropped_entries_shape, context_shape, mask, transform_net_create_fn, *args,
                 unconditional_transform=None, **kwargs):
        super(SurVaeCoupling, self).__init__(mask, transform_net_create_fn,
                                             unconditional_transform=unconditional_transform)
        self.drop_mask = torch.ones(self.features, dtype=torch.bool)
        feature_to_drop = self.identity_features[-1]
        self.drop_mask[feature_to_drop] = 0
        _, self.sorted_indices = torch.sort(
            torch.cat((torch.arange(self.features)[self.drop_mask], torch.tensor([feature_to_drop]))))

        # A flow that can generate one dropped index of the input data given the other data entries
        self.generator = make_generator(dropped_entries_shape, context_shape)

    def forward(self, inputs, context=None):
        faux_output, log_contr = super().forward(inputs, context=context)
        output = faux_output[:, self.drop_mask, ...]
        likelihood = self.generator.log_prob(faux_output[:, ~self.drop_mask, ...], context=output)
        # return output, log_contr + likelihood
        return output, log_contr + likelihood

    def inverse(self, inputs, context=None):
        input_dropped = self.generator.sample(1, context=inputs)
        likelihood = self.generator.log_prob(input_dropped, context=inputs)
        inputs = torch.cat((inputs, input_dropped), 1)[:, self.sorted_indices, ...]
        output, log_contr = super().inverse(inputs, context=context)
        return output, log_contr + likelihood


class wrap_rqct(transforms.PiecewiseRationalQuadraticCouplingTransform):

    def __init__(
            self,
            *args,
            mask=None,
            transform_net_create_fn=None,
            num_bins=10,
            tails=None,
            tail_bound=1.0,
            apply_unconditional_transform=False,
            img_shape=None,
            min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
            min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
            min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE,
            **kwargs
    ):
        super(wrap_rqct, self).__init__(
            self,
            mask,
            transform_net_create_fn,
            # TODO: why isn't this accepted?
            # num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            apply_unconditional_transform=apply_unconditional_transform,
            img_shape=img_shape,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative
        )


# class surRqNSF(wrap_rqct, SurVaeCoupling):

# def __init__(self, *args, **kwargs):
#     super(surRqNSF, self).__init__(*args, **kwargs)

# def __init__(self, dropped_entries_shape, context_shape, *args, **kwargs):
# transforms.PiecewiseRationalQuadraticCouplingTransform.__init__(self, *args, **kwargs)
# SurVaeCoupling.__init__(self, dropped_entries_shape, context_shape, *args, **kwargs)

class surRqNSF(transforms.PiecewiseRationalQuadraticCouplingTransform):
    def __init__(self, dropped_entries_shape, context_shape, mask, transform_net_create_fn,
                 num_bins=10,
                 tails=None,
                 tail_bound=1.0,
                 apply_unconditional_transform=False,
                 img_shape=None,
                 min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
                 min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE,
                 generator_function=create_flow):
        super(surRqNSF, self).__init__(mask=mask,
                                       transform_net_create_fn=transform_net_create_fn,
                                       tails=tails,
                                       tail_bound=tail_bound,
                                       num_bins=num_bins,
                                       apply_unconditional_transform=apply_unconditional_transform,
                                       min_bin_width=min_bin_width,
                                       min_bin_height=min_bin_height,
                                       min_derivative=min_derivative,
                                       img_shape=img_shape
                                       )
        self.drop_mask = torch.ones(self.features, dtype=torch.bool)
        feature_to_drop = self.identity_features[-1]
        self.drop_mask[feature_to_drop] = 0
        _, self.sorted_indices = torch.sort(
            torch.cat((torch.arange(self.features)[self.drop_mask], torch.tensor([feature_to_drop]))))

        # A flow that can generate one dropped index of the input data given the other data entries
        self.generator = generator_function(dropped_entries_shape, context_shape)
        # self.generator = generator_function(dropped_entries_shape, context_channels=context_shape)

    def forward(self, inputs, context=None):
        faux_output, log_contr = super().forward(inputs, context=context)
        output = faux_output[:, self.drop_mask, ...]
        likelihood = self.generator.log_prob(faux_output[:, ~self.drop_mask, ...], context=output)
        # return output, log_contr + likelihood
        return output, log_contr + likelihood

    def inverse(self, inputs, context=None):
        input_dropped = self.generator.sample(1, context=inputs).squeeze().unsqueeze(1)
        likelihood = self.generator.log_prob(input_dropped, context=inputs)
        inputs = torch.cat((inputs, input_dropped), 1)[:, self.sorted_indices, ...]
        output, log_contr = super().inverse(inputs, context=context)
        # return output, log_contr + likelihood
        # TODO: return the correct likelihood
        return output, inputs.new_zeros(inputs.shape[0]).to(inputs.device)


class BaseCouplingFunnelAlt(nn.Module):
    def __init__(self, coupling_inn, context_shape, dropped_entries_shape, generator_function, **kwargs):
        super(BaseCouplingFunnelAlt, self).__init__()
        self.coupling_inn = coupling_inn
        self.features = coupling_inn.features
        self.keep_mask = torch.zeros(self.features, dtype=torch.bool)
        feature_to_keep = self.coupling_inn.transform_features
        self.keep_mask[feature_to_keep] = 1
        _, self.sorted_indices = torch.sort(
            torch.cat((torch.arange(self.features)[~self.keep_mask], feature_to_keep)))

        # A flow that can generate one dropped index of the input data given the other data entries
        self.generator = generator_function(dropped_entries_shape, context_shape)

    def forward(self, inputs, context=None):
        faux_output, log_contr = self.coupling_inn.forward(inputs, context=context)
        output = faux_output[:, self.keep_mask, ...]
        likelihood = self.generator.log_prob(faux_output[:, ~self.keep_mask, ...], context=output)
        return output, log_contr + likelihood

    def inverse(self, inputs, context=None):
        input_dropped, likelihood = self.generator.sample_and_log_prob(1, context=inputs)
        input_dropped = input_dropped.squeeze()
        inn_input = torch.cat((inputs, input_dropped), 1)[:, self.sorted_indices, ...]
        output, log_contr = self.coupling_inn.inverse(inn_input, context=context)
        return output, log_contr + likelihood.squeeze()


class BaseCouplingFunnel(nn.Module):
    def __init__(self, coupling_inn, context_shape, dropped_entries_shape, generator_function, **kwargs):
        super(BaseCouplingFunnel, self).__init__()
        self.coupling_inn = coupling_inn
        self.features = coupling_inn.features
        self.drop_mask = torch.ones(self.features, dtype=torch.bool)
        feature_to_drop = self.coupling_inn.identity_features
        self.drop_mask[feature_to_drop] = 0
        _, self.sorted_indices = torch.sort(
            torch.cat((torch.arange(self.features)[self.drop_mask], feature_to_drop)))

        # A flow that can generate one dropped index of the input data given the other data entries
        self.generator = generator_function(dropped_entries_shape, context_shape)

    def forward(self, inputs, context=None):
        faux_output, log_contr = self.coupling_inn.forward(inputs, context=context)
        output = faux_output[:, self.drop_mask, ...]
        likelihood = self.generator.log_prob(faux_output[:, ~self.drop_mask, ...], context=output)
        return output, log_contr + likelihood

    def inverse(self, inputs, context=None):
        input_dropped, likelihood = self.generator.sample_and_log_prob(1, context=inputs)
        input_dropped = input_dropped.squeeze()
        if input_dropped.dim() == 1:
            input_dropped = input_dropped.unsqueeze(1)
        inputs = torch.cat((inputs, input_dropped), 1)[:, self.sorted_indices, ...]
        output, log_contr = self.coupling_inn.inverse(inputs, context=context)
        return output, log_contr + likelihood.squeeze()

    # def forward(self, inputs, context=None):
    #     to_keep = inputs[:, :-1, ...]
    #     to_drop = inputs[:, -1, ...].view(-1, 1)
    #     output, log_contr = self.coupling_inn.forward(to_keep, context=to_drop)
    #     likelihood = self.generator.log_prob(to_drop, context=to_keep)
    #     return output, log_contr + likelihood
    #
    # def inverse(self, inputs, context=None):
    #     input_dropped, likelihood = self.generator.sample_and_log_prob(1, context=inputs)
    #     input_dropped = input_dropped.squeeze().unsqueeze(1)
    #     faux_output, log_contr = self.coupling_inn.inverse(inputs, context=input_dropped)
    #     output = torch.cat((faux_output, input_dropped), 1)
    #     return output, log_contr + likelihood.squeeze()


class BaseAutoregressiveFunnel(nn.Module):
    def __init__(self, autoregressive_inn, autoregressive_inn_kwargs, n_drop, generator_function,
                 generator_kwargs=None):
        super(BaseAutoregressiveFunnel, self).__init__()
        if generator_kwargs is None:
            generator_kwargs = {}
        autoregressive_inn_kwargs['features'] -= n_drop
        autoregressive_inn_kwargs['context_features'] = n_drop
        self.autoregressive_inn = autoregressive_inn(**autoregressive_inn_kwargs)
        self.n_drop = n_drop
        # A flow that can generate one dropped index of the input data given the other data entries
        self.generator = generator_function(n_drop, autoregressive_inn_kwargs['features'], **generator_kwargs)

    def arg_missing_exception(self, arg):
        raise Exception('Need to pass {}')

    def forward(self, inputs, context=None):
        dropped_feature = inputs[..., -self.n_drop:]
        inputs = inputs[..., :-self.n_drop]
        output, log_contr = self.autoregressive_inn.forward(inputs, context=dropped_feature.view(-1, self.n_drop))
        likelihood = self.generator.log_prob(dropped_feature, context=output)
        return output, log_contr + likelihood

    def inverse(self, inputs, context=None):
        input_dropped, likelihood = self.generator.sample_and_log_prob(1, context=inputs)
        input_dropped = input_dropped.squeeze()
        if input_dropped.dim == 1:
            input_dropped = input_dropped.unsqueeze(1)
        output_minus, log_contr = self.autoregressive_inn.inverse(inputs, context=input_dropped)
        output = torch.cat((output_minus, input_dropped), -1)
        return output, log_contr + likelihood.squeeze()
