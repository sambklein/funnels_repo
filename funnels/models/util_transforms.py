from nflows.transforms import SqueezeTransform, Transform
import torch
import torch.nn.functional as F
from torch import nn


class ReverseSqueezeTransform(SqueezeTransform):

    def get_output_shape(self, c, h, w):
        return (c // self.factor ** 2, h * self.factor, w * self.factor)

    def forward(self, inputs, context=None):
        return super(ReverseSqueezeTransform, self).inverse(inputs, context)

    def inverse(self, inputs, context=None):
        return super(ReverseSqueezeTransform, self).forward(inputs, context)


class PaddingSurjection(Transform):
    """
    This is a very specific kind of padding operation, essentially just implemented for one use case.
    """

    def __init__(self, pad=0):
        super(PaddingSurjection, self).__init__()
        self.register_buffer('pad', torch.tensor(pad, dtype=torch.int32))

    def get_ldj(self, inputs):
        return torch.zeros(inputs.shape[0])

    def forward(self, inputs, context=None):
        output = torch.nn.functional.pad(inputs, (0, self.pad, 0, self.pad))
        ldj = self.get_ldj(inputs)
        return output, ldj

    def inverse(self, inputs, context=None):
        return inputs[..., :-1, :-1], -self.get_ldj(inputs)


# From https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes * stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar


class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=10, nc=3, w=32):
        super().__init__()
        fact = int(w / 32)
        self.fact = fact
        self.in_planes = 256 * fact

        self.linear = nn.Linear(z_dim, 256 * fact)

        self.layer4 = self._make_layer(BasicBlockDec, 128 * fact, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 64 * fact, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 32 * fact, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 32 * fact, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(32 * fact, nc, kernel_size=3, scale_factor=2)
        # self.conv2 = ResizeConv2d(32, nc, kernel_size=3, scale_factor=2)

        self.register_parameter(name='log_var', param=nn.Parameter(torch.tensor([0.0] * nc)))

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 256 * self.fact, 1, 1)
        x = F.interpolate(x, scale_factor=2 * self.fact)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        mu = torch.sigmoid(self.conv1(x))
        mu = mu.view(mu.size(0), 3, 32 * self.fact, 32 * self.fact)
        # std = torch.sigmoid(self.conv2(x))
        # std = std.view(std.size(0), 3, 32, 32)
        std = self.log_var
        return mu, std


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

def get_model(image_dims, latent_size, direction=1):
    c, h, w = image_dims
    if direction == -1:
        MLP_width = 512

        class decoder(nn.Module):
            def __init__(self):
                super(decoder, self).__init__()
                MLP_layers = [nn.Linear(latent_size, MLP_width),
                              nn.ReLU(),
                              nn.Linear(MLP_width, MLP_width),
                              nn.ReLU(),
                              nn.Linear(MLP_width, MLP_width),
                              nn.ReLU(),
                              nn.Linear(MLP_width, c * h * w),
                              # nn.Sigmoid(),
                              Reshape((c, h, w))]
                self.dec = nn.Sequential(*MLP_layers)
                self.log_scale = nn.Parameter(torch.Tensor([0.0]))

            def forward(self, data):
                return self.dec(data), self.log_scale

        return decoder()
    else:
        MLP_width = 512
        MLP = [Reshape([c * h * w]),
               nn.Linear(c * h * w, MLP_width),
               nn.ReLU(),
               nn.Linear(MLP_width, MLP_width),
               nn.ReLU(),
               nn.Linear(MLP_width, MLP_width),
               nn.ReLU(),
               nn.Linear(MLP_width, latent_size * 2)]
        return MLP