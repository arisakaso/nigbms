# %%
from typing import Dict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


class InterpolateDecoder(torch.nn.Module):
    def __init__(self, out_dim: int = 32, mode="linear"):
        super().__init__()
        self.out_dim = out_dim
        self.mode = mode

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal = signal.unsqueeze(1)
        scale_factor = self.out_dim // signal.shape[-1]
        interpolated_signal = torch.nn.functional.interpolate(
            signal, scale_factor=scale_factor, mode=self.mode, align_corners=True
        )

        return interpolated_signal.squeeze(1)


class InterpolateDecoder2D(torch.nn.Module):
    def __init__(self, out_dim: int = 32, mode="bilinear"):
        super().__init__()
        self.out_dim = out_dim
        self.mode = mode

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        bs, n2 = signal.shape
        n = int(n2**0.5)
        signal = signal.reshape(bs, 1, n, n)
        scale_factor = self.out_dim // n
        interpolated_signal = torch.nn.functional.interpolate(
            signal, scale_factor=scale_factor, mode=self.mode, align_corners=True
        )

        return interpolated_signal.reshape(bs, -1)


class FFTEncoder(torch.nn.Module):
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        freq_signal = torch.fft.rfft(signal, dim=-1)[..., : self.out_dim // 2]
        freq_signal = torch.view_as_real(freq_signal).reshape(-1, self.out_dim)
        return freq_signal


class IFFTDecoder(torch.nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, freq_signal: torch.Tensor) -> torch.Tensor:
        n_components = freq_signal.shape[-1] // 2
        freq_signal = torch.view_as_complex(freq_signal.reshape(-1, n_components, 2))
        signal = torch.fft.irfft(freq_signal, n=self.out_dim, dim=-1)
        return signal


class SinDecoder(torch.nn.Module):
    def __init__(self, out_dim: int = 128, in_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.basis = torch.sin(
            torch.arange(1, out_dim + 1).unsqueeze(-1)
            * torch.tensor([i / (out_dim + 1) for i in range(1, out_dim + 1)])
            * torch.pi
        )
        self.basis = self.basis[:, :in_dim].unsqueeze(0).cuda()  # (1, out_dim, n_basis)

    def forward(self, freq_signal: torch.Tensor) -> torch.Tensor:
        # freq_signal: (bs, n_basis)
        signal = torch.matmul(self.basis, freq_signal.unsqueeze(-1))  # (bs, out_dim, 1)

        return signal.squeeze()


class SinEncoder(torch.nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        N = out_dim
        self.basis = (
            torch.sin(
                torch.arange(1, N + 1).reshape(N, 1)
                * torch.tensor([i / (N + 1) for i in range(1, N + 1)])
                * torch.pi
            )
            .reshape(1, N, N)
            .cuda()
        )  # (1, N, N)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        freq_signal = torch.matmul(
            self.basis.transpose(1, 2), signal.reshape(-1, self.out_dim, 1)
        )

        return freq_signal.squeeze()


class MetaSolver(torch.nn.Module):
    def __init__(
        self, params_learn: DictConfig, features: DictConfig, model: DictConfig
    ):
        super().__init__()
        self.params_learn = params_learn
        self.features = features
        self.model = model

    def _get_features(self, tau: Dict):
        features = {}

        if "b" in self.features:
            features["b"] = tau["b"].unsqueeze(-1)

        if "b_freq" in self.features:
            features["b_freq"] = FFTEncoder(self.features.b_freq.dim)(tau["b"])

        if "b_sin" in self.features:
            features["b_sin"] = SinEncoder(self.features.b_sin.dim)(tau["b"])

        if "features" in self.features:
            features["features"] = torch.log(tau["features"])

        return features

    def forward(self, tau: Dict) -> torch.Tensor:
        raise NotImplementedError


class MetaSolverMLP(MetaSolver):
    def __init__(
        self,
        params_learn: DictConfig,
        features: DictConfig,
        model: DictConfig,
    ):
        super().__init__(params_learn, features, model)

        if model.layers[0] is None:
            in_dim = 0
            for v in features.values():
                in_dim += v.dim
            model.layers[0] = in_dim

        if model.layers[-1] is None:
            out_dim = 0
            for v in params_learn.values():
                out_dim += v.dim
            model.layers[-1] = out_dim

        self.model = instantiate(model)

    def forward(self, tau: Dict) -> torch.Tensor:
        bs = tau["A"].shape[0]
        features = self._get_features(tau)
        x = torch.cat([features[k].reshape(bs, -1) for k in self.features], dim=-1)
        theta = self.model(x)
        return theta


class MetaSolverUNet2D(MetaSolver):
    def __init__(
        self,
        params_learn: DictConfig,
        features: DictConfig,
        model: DictConfig,
    ):
        super().__init__(params_learn, features, model)

        self.model = instantiate(model)

    def forward(self, tau: Dict) -> torch.Tensor:
        bs, n2 = tau["b"].shape
        features = self._get_features(tau)
        n = int(n2**0.5)
        x = features["b"].reshape(bs, 1, n, n)
        theta = self.model(x).reshape(bs, -1)
        return theta


# %%
