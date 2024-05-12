import numpy as np
import torch
from hydra.utils import instantiate
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class ThetaConstructor(Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.decoders = {k: instantiate(v.decoder) for k, v in self.params.items() if v.decoder is not None}

    def forward(self, theta: Tensor) -> TensorDict:
        theta_dict = TensorDict({})
        idx = 0
        for k, v in self.params.items():
            if k in self.decoders:
                in_dim = self.decoders[k].in_dim
                param = self.decoders[k](theta[:, idx : idx + in_dim])
            else:
                in_dim = np.prod(v.shape)
                param = theta[:, idx : idx + in_dim]
            theta_dict[k] = param.reshape(-1, *v.shape)
            idx += in_dim
        return theta_dict


class _Decoder(Module):
    def __init__(self, in_dim: int, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class SinDecoder(_Decoder):
    def __init__(self, in_dim: int = 128, out_dim: int = 128):
        super().__init__(in_dim, out_dim)
        self.basis = torch.sin(
            torch.arange(1, out_dim + 1).unsqueeze(-1)
            * torch.tensor([i / (out_dim + 1) for i in range(1, out_dim + 1)])
            * torch.pi
        )
        self.basis = self.basis[:, :in_dim].unsqueeze(0)  # (1, out_dim, n_basis)
        self.basis = self.basis.cuda()

    def forward(self, theta: Tensor) -> Tensor:
        decoded_theta = torch.matmul(self.basis, theta.unsqueeze(-1))  # (bs, out_dim, 1)
        return decoded_theta.squeeze()


class InterpolateDecoder(Module):
    def __init__(self, out_dim: int = 32, mode="linear"):
        super().__init__()
        self.out_dim = out_dim
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        scale_factor = self.out_dim // x.shape[-1]
        interpolated_signal = F.interpolate(x, scale_factor=scale_factor, mode=self.mode, align_corners=True)

        return interpolated_signal.squeeze(1)


class InterpolateDecoder2D(Module):
    def __init__(self, out_dim: int = 32, mode="bilinear"):
        super().__init__()
        self.out_dim = out_dim
        self.mode = mode

    def forward(self, signal: Tensor) -> Tensor:
        bs, n2 = signal.shape
        n = int(n2**0.5)
        signal = signal.reshape(bs, 1, n, n)
        scale_factor = self.out_dim // n
        interpolated_signal = F.interpolate(signal, scale_factor=scale_factor, mode=self.mode, align_corners=True)

        return interpolated_signal.reshape(bs, -1)


class FFTEncoder(Module):
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, signal: Tensor) -> Tensor:
        freq_signal = torch.fft.rfft(signal, dim=-1)[..., : self.out_dim // 2]
        freq_signal = torch.view_as_real(freq_signal).reshape(-1, self.out_dim)
        return freq_signal


class IFFTDecoder(Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, freq_signal: Tensor) -> Tensor:
        n_components = freq_signal.shape[-1] // 2
        freq_signal = torch.view_as_complex(freq_signal.reshape(-1, n_components, 2))
        signal = torch.fft.irfft(freq_signal, n=self.out_dim, dim=-1)
        return signal


class SinEncoder(Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        N = out_dim
        self.basis = (
            torch.sin(
                torch.arange(1, N + 1).reshape(N, 1) * torch.tensor([i / (N + 1) for i in range(1, N + 1)]) * torch.pi
            )
            .reshape(1, N, N)
            .cuda()
        )  # (1, N, N)

    def forward(self, signal: Tensor) -> Tensor:
        freq_signal = torch.matmul(self.basis.transpose(1, 2), signal.reshape(-1, self.out_dim, 1))

        return freq_signal.squeeze()
