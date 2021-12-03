from copy import deepcopy
import torch
import torch.nn as nn


class XuLinear(nn.Linear):
    def __init__(self, *args, bias=True, **kwargs):
        super(XuLinear, self).__init__(*args, bias=bias, **kwargs)
        torch.nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias.data.fill_(0.0)


class dense_net(nn.Module):
    def __init__(self, input_dim, latent_dim, islast=True, output_activ=nn.Identity(), layers=[300, 300, 300], drp=0,
                 batch_norm=False, layer_norm=False, int_activ=torch.relu, bias=True, vae=False):
        super(dense_net, self).__init__()
        layers = deepcopy(layers)
        # If adding additional layers to the encoder, don't compress directly to the latent dimension
        # Useful when expanind the capacity of these base models to compare with implicit approach

        self.latent_dim = latent_dim
        self.drp_p = drp
        self.inner_activ = int_activ
        self.vae = vae

        self.functions = nn.ModuleList([nn.Linear(input_dim, layers[0], bias=bias)])
        self.functions.extend(
            nn.ModuleList([nn.Linear(layers[i], layers[i + 1], bias=bias) for i in range(len(layers) - 1)]))
        self.final = nn.Linear(layers[-1], latent_dim, bias=bias)
        if self.vae:
            self.log_var = nn.Linear(layers[-1], latent_dim, bias=bias)

        self.output_activ = output_activ

        self.norm = 0
        self.norm_func = nn.LayerNorm
        if batch_norm:
            self.norm = 1
            self.norm_func = nn.BatchNorm1d
        if layer_norm:
            self.norm = 1
            self.norm_func = nn.LayerNorm
        self.norm_funcs = nn.ModuleList([self.norm_func(layers[i]) for i in range(len(layers))])

    def forward(self, x, context=None, **kwargs):
        for i, function in enumerate(self.functions):
            x = function(x)
            if self.norm:
                x = self.norm_funcs[i](x)
            x = self.inner_activ(x)
            x = nn.Dropout(p=self.drp_p)(x)
        y = self.output_activ(self.final(x))
        if self.vae:
            y = y, self.log_var(x)
        return y

    def batch_predict(self, data_array, encode=False):
        store = []
        for data in data_array:
            if encode:
                store += [torch.cat(self.encode(data), 1)]
            else:
                store += [self(data)]
        return torch.cat(store)
