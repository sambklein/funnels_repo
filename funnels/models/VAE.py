import torch
from nflows.distributions import StandardNormal
from torch import nn
import numpy as np

from funnels.models.nn.MLPs import dense_net


class transform(nn.Module):

    def __init__(self, encoder, decoder, latent_size):
        super(transform, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size

    def forward(self, data):
        z_mean, z_log_sigma = self.encoder(data)
        epsilon = torch.normal(mean=torch.zeros_like(z_mean), std=torch.ones_like(z_mean))
        return z_mean + torch.exp(z_log_sigma) * epsilon

    def inverse(self, data):
        x_prime, log_var = self.decoder(data)
        return x_prime, torch.zeros(data.shape[0])


# TODO: clean up the mess that this class has become
class VAE(nn.Module):

    def __init__(self, input_dim, latent_size, layers, activation=torch.relu, encoder=None, decoder=None,
                 preprocess=None, dropout=0.0, batch_norm=False, layer_norm=False):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.preprocess = preprocess
        if encoder is None:
            self.encoder = dense_net(input_dim, latent_size, layers=layers, int_activ=activation, drp=dropout,
                                     vae=True, batch_norm=batch_norm, layer_norm=layer_norm)
        else:
            self.encoder = encoder
        if decoder is None:
            self.decoder = dense_net(latent_size, input_dim, layers=layers, int_activ=activation, drp=dropout, vae=True,
                                     batch_norm=batch_norm, layer_norm=layer_norm)
        else:
            self.decoder = decoder
        self.recon_loss = nn.MSELoss(reduce=False)
        self._transform = transform(self.encoder, self.decoder, latent_size)

    def transform_to_noise(self, data):
        return self._transform(data)

    def set_preprocessing(self, preprocess):
        self.preprocess = preprocess

    def get_scale(self, log_scale):
        scale = log_scale.exp()
        if len(scale) == 3:
            scale = scale.unsqueeze(0).repeat(32, 32, 1).transpose(2, 0)
        return scale

    def likelihood(self, x_prime, log_scale, x):
        scale = self.get_scale(log_scale)
        return torch.distributions.Normal(x_prime, scale).log_prob(x).view(x.shape[0], -1).sum(-1)

    def kl_divergence(self, z, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def get_encoding(self, data):
        enc = self.encoder(data)
        if not isinstance(enc, tuple):
            z_mean, z_log_sigma = enc.split(self.latent_size, dim=1)
        else:
            z_mean, z_log_sigma = enc
        return z_mean, z_log_sigma

    def log_prob(self, data):
        if self.preprocess is not None:
            data, prep_likelihood = self.preprocess(data)
        else:
            prep_likelihood = 0

        z_mean, z_log_sigma = self.get_encoding(data)
        std = (z_log_sigma / 2).exp()
        q = torch.distributions.Normal(z_mean, std)
        z = q.rsample()
        kl_loss = self.kl_divergence(z, z_mean, std)

        x_prime, log_scale = self.decoder(z)
        per_sample_likelihood = self.likelihood(x_prime, log_scale, data + 0.5)

        return per_sample_likelihood - kl_loss + prep_likelihood

    def autoencode(self, data):
        data, prep_likelihood = self.preprocess(data)
        z_mean, z_log_sigma = self.get_encoding(data)
        return self.sample_dec([z_mean, (z_log_sigma / 2).exp()])

    def forward(self, *args):
        raise RuntimeError("This implementation is not intended for use in this fashion.")

    def base_dist_sample(self, encoding):
        z_mean, z_log_sigma = encoding
        epsilon = torch.normal(mean=torch.zeros_like(z_mean), std=torch.ones_like(z_mean))
        return z_mean + torch.exp(z_log_sigma) * epsilon

    def decode(self, sample):
        mu, log_scale = self.decoder(sample)
        std = self.get_scale(log_scale)
        return torch.distributions.Normal(mu, std).sample([1])[0]

    def sample_dec(self, encoding):
        # return self.decoder(self.base_dist_sample(encoding))
        return self.decode(self.base_dist_sample(encoding))

    def sample(self, num, temperature=1):
        return self.decode(StandardNormal([self.latent_size]).sample(num) * temperature)
