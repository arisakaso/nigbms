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
    def __init__(
        self,
        in_dim,  # input dimension
        out_dim,  # output dimension
        n_layers,  # number of hidden layers
        n_units,  # number of neurons in the hidden layers
        n_components,  # number of components
        hidden_activation,
        output_activation,
        batch_normalization,
        init_scale=1.0,
        init_weight=None,
        dropout=0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.decay_rates = nn.Parameter(torch.rand(n_components, 1) * init_scale)  # (n_components, 1)
        self.roll_out = nn.Parameter(torch.arange(out_dim).unsqueeze(0), requires_grad=False)  # (1, out_dim)

        self.layers = nn.Sequential()  # output: (bs, n_components)

        # input layer
        self.layers.append(nn.Linear(in_dim, n_units))
        if batch_normalization:
            self.layers.append(nn.BatchNorm1d(n_units, track_running_stats=False))
        self.layers.append(hidden_activation)
        self.layers.append(nn.Dropout(p=dropout))

        # hidden layers
        for _ in range(n_layers):
            self.layers.append(nn.Linear(n_units, n_units))
            if batch_normalization:
                self.layers.append(nn.BatchNorm1d(n_units, track_running_stats=False))
            self.layers.append(hidden_activation)
            self.layers.append(nn.Dropout(p=dropout))

        # output layer
        self.layers.append(nn.Linear(n_units, n_components))
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
        c = self.layers(x)
        # base = torch.exp(-self.decay_rates * self.roll_out)  # (n_components, out_dim)
        base = torch.pow(self.decay_rates, self.roll_out)  # (n_components, out_dim)
        y = c @ base  # (bs, out_dim)
        return y


class MLP(Module):
    def __init__(
        self,
        in_dim,  # input dimension
        out_dim,  # output dimension
        n_layers,  # number of hidden layers
        n_units,  # number of neurons in the hidden layers
        hidden_activation,
        output_activation,
        batch_normalization,
        init_weight=None,
        dropout=0,
        **kwargs,
    ):
        super(MLP, self).__init__()
        self.out_dim = out_dim

        self.layers = nn.Sequential()

        # input layer
        self.layers.append(nn.Linear(in_dim, n_units))
        if batch_normalization:
            self.layers.append(nn.BatchNorm1d(n_units, track_running_stats=False))
        self.layers.append(hidden_activation)
        self.layers.append(nn.Dropout(p=dropout))

        # hidden layers
        for _ in range(n_layers):
            self.layers.append(nn.Linear(n_units, n_units))
            if batch_normalization:
                self.layers.append(nn.BatchNorm1d(n_units, track_running_stats=False))
            self.layers.append(hidden_activation)
            self.layers.append(nn.Dropout(p=dropout))

        # output layer
        self.layers.append(nn.Linear(n_units, out_dim))
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
        out_dim=1,
        base_channels=64,
        kernel_size=3,
        batch_normalization=False,
        hidden_activation=nn.GELU(),
        output_activation=nn.Identity(),
        n_conv_layers=2,
        n_fcn_layers=2,
        **kwargs,
    ):
        super().__init__()

        self.layers = nn.Sequential()
        self.layers.append(nn.Conv1d(in_channels, base_channels, kernel_size, padding="same"))

        for _ in range(n_conv_layers - 1):
            self.layers.append(nn.Conv1d(base_channels, base_channels, kernel_size, padding="same"))
            if batch_normalization:
                self.layers.append(nn.BatchNorm1d(base_channels, track_running_stats=False))
            self.layers.append(hidden_activation)

        self.layers.append(nn.AdaptiveAvgPool1d(1))
        self.layers.append(nn.Flatten())

        for _ in range(n_fcn_layers - 1):
            self.layers.append(nn.Linear(base_channels, base_channels))
            self.layers.append(hidden_activation)

        self.layers.append(nn.Linear(base_channels, out_dim))
        self.layers.append(output_activation)

    def forward(self, x):
        return self.layers(x)


class CNN1DRes(nn.Module):
    def __init__(
        self,
        in_channels,
        out_dim=1,
        base_channels=64,
        kernel_size=3,
        batch_normalization=False,
        hidden_activation=nn.GELU(),
        output_activation=nn.Identity(),
        n_conv_layers=2,
        n_fcn_layers=2,
        **kwargs,
    ):
        super().__init__()

        self.layers = nn.Sequential()
        self.layers.append(nn.Conv1d(in_channels, base_channels, kernel_size, padding="same"))
        self.layers.append(ResBlock1D(base_channels, kernel_size, hidden_activation))

        for _ in range(n_conv_layers - 1):
            self.layers.append(ResBlock1D(base_channels, kernel_size, hidden_activation))

        self.layers.append(nn.AdaptiveAvgPool1d(1))
        self.layers.append(nn.Flatten())

        for _ in range(n_fcn_layers - 1):
            self.layers.append(nn.Linear(base_channels, base_channels))
            self.layers.append(hidden_activation)

        self.layers.append(nn.Linear(base_channels, out_dim))
        self.layers.append(output_activation)

    def forward(self, x):
        return self.layers(x)


class CNN1DWithPooling(nn.Module):
    def __init__(
        self,
        in_channels,
        out_dim=1,
        base_channels=64,
        kernel_size=3,
        batch_normalization=False,
        hidden_activation=nn.GELU(),
        output_activation=nn.Identity(),
        n_conv_layers=2,
        n_fcn_layers=2,
        **kwargs,
    ):
        super().__init__()

        layers = nn.Sequential()
        layers.append(nn.Conv1d(in_channels, base_channels, kernel_size, padding="same"))
        layers.append(nn.Conv1d(base_channels, base_channels, kernel_size, padding="same"))
        layers.append(hidden_activation)
        layers.append(nn.AvgPool1d(2))

        for i in range(n_conv_layers - 1):  # Double conv
            layers.append(
                nn.Conv1d(base_channels * (2**i), base_channels * (2 ** (i + 1)), kernel_size, padding="same")
            )
            layers.append(
                nn.Conv1d(base_channels * (2 ** (i + 1)), base_channels * (2 ** (i + 1)), kernel_size, padding="same")
            )
            layers.append(hidden_activation)
            layers.append(nn.AvgPool1d(2))

        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())

        # for _ in range(n_fcn_layers - 1):
        #     layers.append(nn.Linear(base_channels, base_channels))
        #     layers.append(hidden_activation)

        layers.append(nn.Linear(base_channels * (2 ** (n_conv_layers - 1)), out_dim))
        layers.append(output_activation)
        self.layers = layers

    def forward(self, x):
        return self.layers(x)


class CNN1DWithPoolingRes(nn.Module):
    def __init__(
        self,
        in_channels,
        out_dim=1,
        base_channels=64,
        kernel_size=3,
        batch_normalization=False,
        hidden_activation=nn.GELU(),
        output_activation=nn.Identity(),
        n_conv_layers=2,
        n_fcn_layers=2,
        **kwargs,
    ):
        super().__init__()

        layers = nn.Sequential()
        layers.append(nn.Conv1d(in_channels, base_channels, kernel_size, padding="same"))
        layers.append(ResBlock1D(base_channels, base_channels))
        layers.append(nn.AvgPool1d(2))

        for i in range(n_conv_layers - 1):  # Double conv
            layers.append(
                nn.Conv1d(base_channels * (2**i), base_channels * (2 ** (i + 1)), kernel_size, padding="same")
            )
            layers.append(ResBlock1D(base_channels * (2 ** (i + 1)), base_channels * (2 ** (i + 1))))
            layers.append(nn.AvgPool1d(2))

        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())

        # for _ in range(n_fcn_layers - 1):
        #     layers.append(nn.Linear(base_channels, base_channels))
        #     layers.append(hidden_activation)

        layers.append(nn.Linear(base_channels * (2 ** (n_conv_layers - 1)), out_dim))
        layers.append(output_activation)
        self.layers = layers

    def forward(self, x):
        return self.layers(x)


class ResBlock1D(nn.Module):
    def __init__(self, n_channels, kernel_size=3, activation=nn.GELU()):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding="same"
        )
        self.bn1 = nn.BatchNorm1d(n_channels, track_running_stats=False)
        self.conv2 = nn.Conv1d(
            in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding="same"
        )
        self.bn2 = nn.BatchNorm1d(n_channels, track_running_stats=False)
        self.activation = activation

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = x + inputs
        return self.activation(out)


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
