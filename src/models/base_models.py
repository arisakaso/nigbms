# %%
import torch
from torch import nn


class Exp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return torch.exp(x)


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.square(x)


class MLP(nn.Module):
    def __init__(
        self,
        layers,
        hidden_activation,
        output_activation,
        batch_normalization,
        dropout=0,
    ):
        super(MLP, self).__init__()

        # Create the fully connected layers using the provided layer sizes
        self.layers = nn.Sequential()

        # hidden layers
        for i in range(len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if batch_normalization:
                self.layers.append(nn.BatchNorm1d(layers[i + 1]))
            self.layers.append(eval(hidden_activation)())
            self.layers.append(nn.Dropout(p=dropout))

        # output layer
        self.layers.append(nn.Linear(layers[-2], layers[-1]))
        self.layers.append(eval(output_activation)())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLPSkip(nn.Module):
    def __init__(
        self, layers, hidden_activation, output_activation, batch_normalization
    ):
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
        self.out_layer = nn.Sequential(
            nn.Linear(layers[-2], layers[-1]), eval(output_activation)()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = []

        for block in self.blocks:
            if (
                x.shape[1] == block[0].out_features
            ):  # Check for possible skip connection
                residuals.append(x)

            x = block(x)

            if (
                residuals and x.shape[1] == residuals[-1].shape[1]
            ):  # If sizes match for skip connection
                x += residuals.pop()

        x = self.out_layer(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        batch_normalization=False,
        activation="torch.nn.GELU",
    ):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.activation = eval(activation)()
        self.batch_normalization = batch_normalization

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_normalization:
            x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        if self.batch_normalization:
            x = self.bn2(x)
        x = self.activation(x)

        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, kernel_size=3):
        super().__init__()

        # Encoder
        self.encoder_conv1 = ConvBlock(in_channels, base_channels, kernel_size)
        self.encoder_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder_conv2 = ConvBlock(base_channels, base_channels * 2, kernel_size)
        self.encoder_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder_conv3 = ConvBlock(
            base_channels * 2, base_channels * 4, kernel_size
        )
        self.encoder_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = ConvBlock(base_channels * 4, base_channels * 8)

        # Decoder
        self.decoder_upsample3 = nn.ConvTranspose1d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.decoder_conv3 = ConvBlock(
            base_channels * 8, base_channels * 4, kernel_size
        )
        self.decoder_upsample2 = nn.ConvTranspose1d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.decoder_conv2 = ConvBlock(
            base_channels * 4, base_channels * 2, kernel_size
        )
        self.decoder_upsample1 = nn.ConvTranspose1d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.decoder_conv1 = ConvBlock(base_channels * 2, base_channels, kernel_size)

        # Output
        self.output_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_conv1(x)
        x = self.encoder_pool1(x1)

        x2 = self.encoder_conv2(x)
        x = self.encoder_pool2(x2)

        x3 = self.encoder_conv3(x)
        x = self.encoder_pool3(x3)

        # Bottleneck
        x = self.bottleneck_conv(x)

        # Decoder
        x = self.decoder_upsample3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder_conv3(x)

        x = self.decoder_upsample2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder_conv2(x)

        x = self.decoder_upsample1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder_conv1(x)

        # Output
        x = self.output_conv(x)

        return x


class CNN1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_unit=1,
        base_channels=64,
        kernel_size=3,
        batch_normalization=False,
        hidden_activation="torch.nn.GELU",
        output_activation="torch.nn.Identity",
        hidden_unit_factor=1,
    ):
        super().__init__()

        self.pad = nn.ConstantPad1d(3, 0)
        self.conv1 = nn.Conv1d(
            in_channels, base_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.conv2 = nn.Conv1d(
            base_channels, base_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.conv3 = nn.Conv1d(
            base_channels, base_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(base_channels, base_channels * hidden_unit_factor)
        self.fc2 = nn.Linear(base_channels * hidden_unit_factor, out_unit)

        self.activation = eval(hidden_activation)()

    def forward(self, x):
        # Encoder
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        # x = self.conv3(x)
        # x = self.activation(x)

        # x = x.sum(dim=(1,2)).unsqueeze(1)
        x = self.gap(x).squeeze(-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x


# %%
