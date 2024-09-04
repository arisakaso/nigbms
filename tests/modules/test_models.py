import pytest
import torch
import torch.nn as nn
from nigbms.modules.models import CNN1D, MLP, DoubleConv1D, UNet1D


def test_mlp():
    model = MLP()
    assert isinstance(model, MLP)

    batch_size = 5
    x = torch.ones(batch_size, model.in_dim)
    y = model(x)
    assert y.shape == torch.Size([batch_size, model.out_dim])


@pytest.mark.parametrize(
    "in_ch, out_ch, activation, skip, batch_norm, kernel_size, downsample",
    [
        # in_ch == out_ch
        (3, 3, nn.ReLU(), False, True, 3, False),
        (3, 3, nn.ReLU(), True, True, 3, False),
        (3, 3, nn.ReLU(), True, True, 3, True),
        (3, 3, nn.ReLU(), False, False, 3, False),
        (3, 3, nn.ReLU(), False, False, 3, True),
        (3, 3, nn.ReLU(), True, False, 3, True),
        # in_ch != out_ch
        (3, 8, nn.ReLU(), False, True, 3, False),
        (3, 8, nn.ReLU(), True, True, 3, False),
        (3, 8, nn.ReLU(), True, True, 3, True),
        (3, 8, nn.ReLU(), False, False, 3, False),
        (3, 8, nn.ReLU(), False, False, 3, True),
        (3, 8, nn.ReLU(), True, False, 3, True),
    ],
)
def test_double_conv1d(in_ch, out_ch, activation, skip, batch_norm, kernel_size, downsample):
    model = DoubleConv1D(
        in_ch=in_ch,
        out_ch=out_ch,
        activation=activation,
        skip=skip,
        batch_norm=batch_norm,
        kernel_size=kernel_size,
        downsample=downsample,
    )
    assert isinstance(model, DoubleConv1D)

    bs = 2
    length = 64
    input_shape = (bs, in_ch, length)
    x = torch.ones(input_shape)
    output = model(x)

    # Check output shape
    expected_output_shape = (bs, out_ch, length // 2 if downsample else length)

    assert output.shape == expected_output_shape


@pytest.mark.parametrize(
    "in_channels, out_dim, base_channels, kernel_size, n_layers, skip, downsample",
    [
        # Test with different values for in_channels
        (1, 1, 64, 3, 2, False, False),
        (2, 1, 64, 3, 2, False, False),
        (3, 1, 64, 3, 2, False, False),
        # Test with different values for out_dim
        (1, 2, 64, 3, 2, False, False),
        (1, 3, 64, 3, 2, False, False),
        (1, 4, 64, 3, 2, False, False),
        # Test with different values for base_channels
        (1, 1, 32, 3, 2, False, False),
        (1, 1, 128, 3, 2, False, False),
        (1, 1, 256, 3, 2, False, False),
        # Test with different values for kernel_size
        (1, 1, 64, 2, 2, False, False),
        (1, 1, 64, 4, 2, False, False),
        (1, 1, 64, 11, 2, False, False),
        # Test with different values for n_layers
        (1, 1, 64, 3, 1, False, False),
        (1, 1, 64, 3, 3, False, False),
        (1, 1, 64, 3, 4, False, False),
        # Test with different values for skip
        (1, 1, 64, 3, 2, True, False),
        (1, 1, 64, 3, 2, True, True),
        # Test with different values for downsample
        (1, 1, 64, 3, 2, False, True),
        (1, 1, 64, 3, 2, True, True),
    ],
)
def test_cnn1d(in_channels, out_dim, base_channels, kernel_size, n_layers, skip, downsample):
    model = CNN1D(in_channels, out_dim, base_channels, kernel_size, n_layers, skip=skip, downsample=downsample)
    assert isinstance(model, CNN1D)

    bs = 2
    length = 64
    input_shape = (bs, in_channels, length)
    x = torch.ones(input_shape)
    output = model(x)

    # Check output shape
    expected_output_shape = (bs, out_dim)

    assert output.shape == expected_output_shape


def test_unet1d():
    model = UNet1D(in_channels=1, out_channels=1, base_channels=64, n_layers=3)
    assert isinstance(model, UNet1D)

    bs = 2
    length = 64
    input_shape = (bs, 1, length)
    x = torch.ones(input_shape)
    output = model(x)

    # Check output shape
    expected_output_shape = (bs, 1, length)

    assert output.shape == expected_output_shape
