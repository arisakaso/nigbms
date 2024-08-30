import torch
from torch import Tensor, nn
from torch.nn import Module, Parameter
from torch.nn import functional as F

### activation functions


class Exp(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return torch.exp(x)


class Square(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.square(x)


class Constant(Module):
    def __init__(self, shape, range):
        super().__init__()
        theta = torch.distributions.Uniform(*range).sample(shape)
        self.weight = Parameter(theta)

    def forward(self, x):
        return self.weight


class InverseBoxCox(Module):
    def __init__(self, lmbda=0):
        super().__init__()
        self.lmbda = lmbda

    def forward(self, x):
        if self.lmbda == 0:
            return torch.exp(x)
        else:
            return torch.exp(torch.log(self.lmbda * x + 1) / self.lmbda)


class ExponentialDecay(Module):
    def __init__(self, in_dim, out_dim, n_units, n_components, **kwargs) -> None:
        super().__init__()

        self.decay_rates = nn.Parameter(torch.rand(n_components, 1))  # (n_components, 1)
        self.roll_out = nn.Parameter(torch.arange(out_dim).unsqueeze(0), requires_grad=False)  # (1, out_dim)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, n_units),
            nn.GELU(),
            nn.Linear(n_units, n_components),
            nn.ReLU(),  # must be postive
        )  # output: (bs, n_components)

    def forward(self, x: Tensor) -> Tensor:
        c = self.layers(x)
        base = torch.exp(-self.decay_rates * self.roll_out)  # (n_components, out_dim)
        y = c @ base  # (bs, out_dim)
        return y


class MLP(Module):
    def __init__(
        self,
        in_dim,  # input dimension
        out_dim,  # output dimension
        num_layers,  # number of hidden layers
        num_neurons,  # number of neurons in the hidden layers
        hidden_activation,
        output_activation,
        batch_normalization,
        init_weight=None,
        dropout=0,
    ):
        super(MLP, self).__init__()

        self.layers = nn.Sequential()

        # input layer
        self.layers.append(nn.Linear(in_dim, num_neurons))
        if batch_normalization:
            self.layers.append(nn.BatchNorm1d(num_neurons, track_running_stats=False))
        self.layers.append(hidden_activation)
        self.layers.append(nn.Dropout(p=dropout))

        # hidden layers
        for _ in range(num_layers):
            self.layers.append(nn.Linear(num_neurons, num_neurons))
            if batch_normalization:
                self.layers.append(nn.BatchNorm1d(num_neurons, track_running_stats=False))
            self.layers.append(hidden_activation)
            self.layers.append(nn.Dropout(p=dropout))

        # output layer
        self.layers.append(nn.Linear(num_neurons, out_dim))
        self.layers.append(output_activation)

        # initialize weights
        if init_weight is not None:
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    if init_weight.dist == "normal":
                        nn.init.normal_(layer.weight, mean=0, std=init_weight.scale)
                    elif init_weight.dist == "uniform":
                        nn.init.uniform_(layer.weight, a=-init_weight.scale, b=init_weight.scale)
                        nn.init.uniform_(layer.bias, a=-init_weight.scale, b=init_weight.scale)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class MLPSkip(Module):
    def __init__(self, layers, hidden_activation, output_activation, batch_normalization):
        super(MLPSkip, self).__init__()

        self.blocks = nn.ModuleList()

        for i in range(len(layers) - 2):
            block = []
            # Linear layer
            block.append(nn.Linear(layers[i], layers[i + 1]))

            # Batch normalization
            if batch_normalization:
                block.append(nn.BatchNorm1d(layers[i + 1]))

            # Activation
            block.append(eval(hidden_activation)())

            # Add the block to the list
            self.blocks.append(nn.Sequential(*block))

        # Output layer
        self.out_layer = nn.Sequential(nn.Linear(layers[-2], layers[-1]), eval(output_activation)())

    def forward(self, x: Tensor) -> Tensor:
        residuals = []

        for block in self.blocks:
            if x.shape[1] == block[0].out_features:  # Check for possible skip connection
                residuals.append(x)

            x = block(x)

            if residuals and x.shape[1] == residuals[-1].shape[1]:  # If sizes match for skip connection
                x += residuals.pop()

        x = self.out_layer(x)

        return x


class CNN1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_unit=1,
        base_channels=64,
        kernel_size=3,
        batch_normalization=False,
        hidden_activation="nn.GELU",
        output_activation="nn.Identity",
        num_conv_layers=2,
        num_fcn_layers=2,
    ):
        super().__init__()

        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels, base_channels, kernel_size, padding="same")])
        self.fcn_layers = nn.ModuleList()

        for _ in range(num_conv_layers - 1):
            self.conv_layers.append(nn.Conv1d(base_channels, base_channels, kernel_size, padding="same"))
            if batch_normalization:
                self.conv_layers.append(nn.BatchNorm1d(base_channels, track_running_stats=False))
            self.conv_layers.append(eval(hidden_activation)())

        self.gap = nn.AdaptiveAvgPool1d(1)

        for _ in range(num_fcn_layers - 1):
            self.fcn_layers.append(nn.Linear(base_channels, base_channels))
            self.fcn_layers.append(eval(hidden_activation)())

        self.fcn_last = nn.Linear(base_channels, out_unit)

        self.output_activation = eval(output_activation)()

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.gap(x).squeeze(-1)

        for layer in self.fcn_layers:
            x = layer(x)

        x = self.fcn_last(x)

        return self.output_activation(x)


class UNet(nn.Module):
    """Pytorch Lightning implementation of U-Net.

    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_

    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

    Implemented by:

        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_

    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.

    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class DoubleConv(nn.Module):
    """[ Conv2d => BatchNorm => ReLU ] x 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature map
    from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False) -> None:
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


### UNet1D


class UNet1D(nn.Module):
    """UNet 1D implementation."""

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv1D(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down1D(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up1D(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv1d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1 : self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class DoubleConv1D(nn.Module):
    """[ Conv1d => BatchNorm => ReLU ] x 2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Down1D(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool1d(kernel_size=2, stride=2), DoubleConv1D(in_ch, out_ch))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Up1D(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature map
    from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False) -> None:
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv1d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose1d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv1D(in_ch, out_ch)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff = x2.shape[2] - x1.shape[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
