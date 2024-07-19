# %%
from typing import Dict

import torch
import torch.autograd.forward_ad as fwAD
from hydra.utils import instantiate
from torch import Tensor
from torch.autograd import grad

from src.models.meta_solvers import FFTEncoder, SinEncoder
from src.solvers.pytorch import Solver
from src.utils.utils import extract_param


# %%
class SurrogateSolver(Solver):
    def __init__(self, params_fix: Dict, params_learn: Dict, features: Dict) -> None:
        super().__init__(params_fix, params_learn)
        self.features = features

    def _preprocess(self, tau, theta):
        raise NotImplementedError

    def _get_features(self, tau, theta):
        """_get_features

        _extended_summary_

        Args:
            tau (_type_): task
            theta (_type_): (decoded) parameters

        Returns:
            features
        """
        features = {}
        b = tau["b"].unsqueeze(-1)
        A = tau["A"]
        x0 = extract_param("x0", self.params_learn, theta).unsqueeze(-1)

        if "x_sol" in self.features:
            features["x_sol"] = tau["x_sol"]
        if "x_sol_freq" in self.features:
            n = self.features.x_sol_freq.dim
            features["x_sol_freq"] = FFTEncoder(n)(tau["x_sol"])

        if "b" in self.features:
            features["b"] = tau["b"]
        if "b_freq" in self.features:
            n = self.features.b_freq.dim
            features["b_freq"] = FFTEncoder(n)(tau["b"])

        if "x0" in self.features:
            features["x0"] = x0.squeeze(-1)

        if "x0_freq" in self.features:
            n = self.features.x0_freq.dim
            features["x0_freq"] = FFTEncoder(n)(x0.squeeze())

        if "r0" in self.features:
            r0 = b - A @ x0
            features["r0"] = r0.squeeze(-1)

        if "e0" in self.features:
            e0 = x0 - tau["x_sol"].unsqueeze(-1)
            features["e0"] = e0.squeeze(-1)

        if "e0_freq" in self.features:
            n = self.features.e0_freq.dim
            x_sol_freq = FFTEncoder(n)(tau["x_sol"])
            x0_freq = FFTEncoder(n)(x0.squeeze())
            features["e0_freq"] = x0_freq - x_sol_freq

        if "e0_freq_abs" in self.features:
            n = self.features.e0_freq_abs.dim
            x_sol_freq = FFTEncoder(n)(tau["x_sol"])
            features["e0_freq_abs"] = torch.abs(x0.squeeze(-1) - x_sol_freq)

        if "e0_sin" in self.features:
            n = self.features.e0_sin.dim
            x_sin = SinEncoder(n)(tau["x_sol"])
            x0_sin = SinEncoder(n)(x0)
            features["e0_sin"] = x0_sin - x_sin

        return features

    def forward(self, tau, theta):
        raise NotImplementedError


class SurrogateSolverMLP(SurrogateSolver):
    def __init__(
        self, params_fix: Dict, params_learn: Dict, features: Dict, model: Dict
    ) -> None:
        super().__init__(params_fix, params_learn, features)

        in_dim = 0
        for v in features.values():
            in_dim += v["dim"]
        model.layers[0] = in_dim
        self.model = instantiate(model)

    def forward(self, tau: Dict, theta: Tensor) -> Tensor:
        features = self._get_features(tau, theta)
        x = torch.cat([features[k] for k in self.features.keys()], dim=1)
        y = self.model(x)
        return y


class SurrogateSolverCNN1D(SurrogateSolver):
    def __init__(
        self, params_fix: Dict, params_learn: Dict, features: Dict, model: Dict
    ) -> None:
        super().__init__(params_fix, params_learn, features)

        model.in_channels = len(features)
        self.model = instantiate(model)

    def _preprocess(self, tau: Dict, theta: Tensor) -> Tensor:
        features = self._get_features(tau, theta)
        inputs = [features[k].reshape(-1, 1, v.dim) for k, v in self.features.items()]
        x = torch.cat(inputs, dim=1)
        return x

    def forward(self, tau: Dict, theta: Tensor) -> Tensor:
        x = self._preprocess(tau, theta)
        y = self.model(x)
        return y


def jvp(f, x: Tensor, v: Tensor, jvp_type: str, eps: float):
    if jvp_type == "forwardAD":
        with fwAD.dual_level():
            dual_input = fwAD.make_dual(x, v)
            dual_output = f(dual_input)
            y, dvf = fwAD.unpack_dual(dual_output)

    elif jvp_type == "forwardFD":
        y = f(x)
        y_plus = f(x + eps * v)
        dvf = (y_plus - y) / eps

    elif jvp_type == "centralFD":
        y = f(x)
        y_plus = f(x + eps * v)
        y_minus = f(x - eps * v)
        dvf = (y_plus - y_minus) / (2 * eps)

    else:
        raise NotImplementedError

    return y, dvf


class register_custom_grad_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, d: Dict):
        """_summary_

        Args:
            x: input
            d: dictionary with keys: y, dvf, y_hat, dvf_hat, grad_type

        Returns:
            y
        """
        ctx.d = d
        ctx.save_for_backward(x, d["v"])

        return d["y"].detach()

    @staticmethod
    def backward(ctx, grad_y):
        x, v = ctx.saved_tensors
        d = ctx.d
        v = v * d["v_scale"]
        bs, in_dim = x.shape

        if d["grad_type"] == "f_true":
            # full gradient of f
            try:
                f_true = grad(
                    d["y"],
                    x,
                    grad_outputs=grad_y,
                    retain_graph=True,
                )[0]
            except Exception:
                f_true = torch.zeros_like(x)

            return f_true, None

        # forward gradient of f
        f_fwd = torch.sum(grad_y * d["dvf"], dim=1, keepdim=True) * v
        f_fwd = f_fwd.reshape(d["Nv"], bs, in_dim).mean(dim=0)  # average over Nv

        if d["grad_type"] == "f_fwd":
            return f_fwd, None

        # forward gradient of f_hat
        f_hat_fwd = torch.sum(grad_y * d["dvf_hat"], dim=1, keepdim=True) * v
        f_hat_fwd = f_hat_fwd.reshape(d["Nv"], bs, in_dim).mean(
            dim=0
        )  # average over Nv

        # full gradient of f_hat
        f_hat_true = (
            grad(d["y_hat"], x, grad_outputs=grad_y, retain_graph=True)[0] / d["Nv"]
        )

        if d["grad_type"] == "f_hat_true":
            return f_hat_true, None

        elif d["grad_type"] == "cv_fwd":
            # control variates
            cv_fwd = f_fwd - (f_hat_fwd - f_hat_true)
            return cv_fwd, None

        else:
            raise NotImplementedError
