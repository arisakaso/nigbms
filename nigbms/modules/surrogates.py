# %%
from typing import Dict

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from hydra.utils import instantiate
from jaxtyping import Float
from src.models.meta_solvers import FFTEncoder, SinEncoder
from src.solvers.pytorch import Solver
from src.solvers.utils import extract_param
from torch import Tensor
from torch.autograd import grad
from torch.nn.functional import normalize


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
            r0 = b - A @ x0  # scaler may be not good
            features["r0"] = r0.squeeze(-1)

        if "e0" in self.features:
            e0 = x0 - tau["x_sol"].unsqueeze(-1)  # * 100  # scaler may be not good
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


class SurrogateSolverHyper(SurrogateSolver):
    def __init__(
        self,
        params_fix: Dict,
        params_learn: Dict,
        features: Dict,
        mnet: Dict,
        hnet: Dict,
    ) -> None:
        super().__init__(params_fix, params_learn, features)
        mnet = instantiate(mnet)
        m_func, m_params = functorch.make_functional(mnet)
        self.mp_shapes = [mp.shape for mp in m_params]
        self.mp_offsets = [0] + list(np.cumsum([mp.numel() for mp in m_params]))
        hnet.layers[-1] = int(self.mp_offsets[-1])
        hnet = instantiate(hnet)
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.SiLU(),
        )
        self.m_func = torch.vmap(m_func)
        self.hnet = hnet

    def generate_params(self, tau, theta):
        features = self._get_features(tau, theta)
        z = torch.cat([features["x_sol"], features["b"], features["x0"].detach().clone()], dim=1)
        params = self.hnet(z)
        self.params_lst = []
        for i, shape in enumerate(self.mp_shapes):
            j0, j1 = self.mp_offsets[i], self.mp_offsets[i + 1]
            self.params_lst.append(params[..., j0:j1].reshape(-1, *shape))

    def forward(self, tau: Dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
        features = self._get_features(tau, theta)
        x = features["x0"]

        # x = self.feature_extractor(x)
        y = self.m_func(self.params_lst, x)
        return y


class SurrogateSolverOperator(SurrogateSolver):
    def __init__(
        self,
        params_fix: Dict,
        params_learn: Dict,
        features: Dict,
        branch: Dict,
        trunk: Dict,
    ) -> None:
        super().__init__(params_fix, params_learn, features)

        self.branch = instantiate(branch)
        self.trunk = instantiate(trunk)

    def forward(self, tau: Dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
        features = self._get_features(tau, theta)
        x = features["x0_freq"]
        z = torch.cat([features["x_sol_freq"], features["b_freq"]], dim=1)
        b = self.branch(z)
        t = self.trunk(x)
        y = torch.sum(b * t, dim=1, keepdim=True)
        return y


class SurrogateSolverMLP(SurrogateSolver):
    def __init__(self, params_fix: Dict, params_learn: Dict, features: Dict, model: Dict) -> None:
        super().__init__(params_fix, params_learn, features)

        in_dim = 0
        for v in features.values():
            in_dim += v["dim"]
        model.layers[0] = in_dim
        self.model = instantiate(model)

    def forward(self, tau: Dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
        features = self._get_features(tau, theta)
        x = torch.cat([features[k] for k in self.features.keys()], dim=1)
        y = self.model(x)
        return y


class SurrogateSolverCNN1D(SurrogateSolver):
    def __init__(self, params_fix: Dict, params_learn: Dict, features: Dict, model: Dict) -> None:
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


class SurrogateSolverTransformer(SurrogateSolver):
    def __init__(self, params_fix: Dict, params_learn: Dict, features: Dict, model: Dict) -> None:
        super().__init__(params_fix, params_learn, features)
        self.model = torch.nn.Sequential(
            torch.nn.TransformerEncoderLayer(
                d_model=4,
                nhead=4,
                dim_feedforward=128,
                dropout=0,
                activation="gelu",
                batch_first=True,
            ),
            torch.nn.TransformerEncoderLayer(
                d_model=4,
                nhead=4,
                dim_feedforward=128,
                dropout=0,
                activation="gelu",
                batch_first=True,
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 4, 1),
        )

    def _preprocess(self, tau: Dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs _"]:
        features = self._get_features(tau, theta)
        inputs = [features[k].reshape(-1, v.dim, 1) for k, v in self.features.items()]
        x = torch.cat(inputs, dim=-1)  # need (bs, length, features)
        return x

    def forward(self, tau: Dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
        x = self._preprocess(tau, theta)
        y = self.model(x)
        return y


class SurrogateSolverIterative(SurrogateSolver):
    def __init__(
        self,
        params_fix: Dict,
        params_learn: Dict,
        features: Dict,
        encoder: Dict,
        iterative_func: Dict,
        iterations: int,
    ) -> None:
        super().__init__(params_fix, params_learn, features)
        in_dim = 0
        for v in features.values():
            in_dim += v["dim"]
        encoder.layers[0] = in_dim
        iterative_func.layers[0] = encoder.layers[-1] + 1  # add index
        iterative_func.layers[-1] = encoder.layers[-1]
        self.encoder = instantiate(encoder)
        self.iterative_func = instantiate(iterative_func)
        self.readout = torch.nn.Linear(iterative_func.layers[-1] + 1, 1)
        self.iterations = iterations

    def _preprocess(self, tau: Dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs _"]:
        features = self._get_features(tau, theta)
        inputs = [features[k].reshape(-1, v.dim) for k, v in self.features.items()]
        x = torch.cat(inputs, dim=-1)
        return x, features

    def forward(self, tau: Dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
        x, features = self._preprocess(tau, theta)
        ref = features["e0"].norm(dim=1, keepdim=True)
        enc = self.encoder(x)
        idx = torch.linspace(0, 1, self.iterations + 1, device=x.device).repeat(x.shape[0], 1)  # step index
        enc = torch.cat([enc, idx[:, [0]]], dim=-1)
        outputs = []
        for i in range(self.iterations):
            enc = self.iterative_func(enc)
            enc = torch.cat([enc, idx[:, [i + 1]]], dim=-1)
            outputs.append(self.readout(enc) + ref)
        y = torch.stack(outputs, dim=1)
        return y


class SurrogateSolverKrylov(SurrogateSolver):
    def __init__(
        self,
        params_fix: Dict,
        params_learn: Dict,
        features: Dict,
        model: Dict,
        scale=0.9,
    ) -> None:
        super().__init__(params_fix, params_learn, features)

        in_dim = 0
        for v in features.values():
            in_dim += v["dim"]
        model.layers[0] = in_dim
        self.model = instantiate(model)
        self.scale = scale

    def get_krylov_basis(self, tau, theta):
        b = tau["features"]["b"].unsqueeze(-1)
        A = tau["A"]
        x0 = extract_param("x0", self.params_learn, theta).unsqueeze(-1)
        r0 = b - A @ x0
        rn = r0
        basis = [rn]
        A_normalized = A / A.norm(dim=(1, 2), keepdim=True, p=2)
        for i in range(A.shape[1] - 1):
            # rn = normalize(A @ rn) * self.scale**i
            rn = A_normalized @ rn
            basis.append(rn)
        basis = torch.cat(basis, dim=-1)
        return basis

    def _preprocess(self, tau: Dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs _"]:
        features = self._get_features(tau, theta)
        inputs = [features[k].reshape(-1, v.dim) for k, v in self.features.items()]
        x = torch.cat(inputs, dim=-1)
        return x

    def forward(self, tau: Dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
        x = self._preprocess(tau, theta)
        x0 = extract_param("x0", self.params_learn, theta).unsqueeze(-1)
        x_sol = tau["x_sol"].unsqueeze(-1)

        basis = self.get_krylov_basis(tau, theta)  # (bs, dim, iterations)
        coefficients = self.model(x).unsqueeze(1)  # (bs, 1, iterations) coefficients of basis
        components = coefficients * basis  # (bs, dim, iterations)
        history = x0 + torch.cumsum(components, dim=-1)  # (bs, dim, iterations + 1)
        y = (history - x_sol).norm(dim=1)
        # y = history.transpose(1, 2)  # history_solution

        return y


class SurrogateSolverKrylovSeq(SurrogateSolver):
    def __init__(
        self,
        params_fix: Dict,
        params_learn: Dict,
        features: Dict,
        encoder: Dict,
        iterative_func,
    ) -> None:
        super().__init__(params_fix, params_learn, features)

        in_dim = 0
        for v in features.values():
            in_dim += v["dim"]
        encoder.layers[0] = in_dim
        self.encoder = instantiate(encoder)
        self.iterative_func = instantiate(iterative_func)
        self.readout = torch.nn.Linear(iterative_func.layers[-1], 1)

    def get_krylov_basis(self, tau, theta):
        b = tau["features"]["b"].unsqueeze(-1)
        A = tau["A"]
        x0 = extract_param("x0", self.params_learn, theta).unsqueeze(-1)
        r0 = b - A @ x0
        rn = normalize(r0)
        basis = [rn]
        A_normalized = A / A.norm(dim=(1, 2), keepdim=True, p=2)
        for i in range(A.shape[1] - 1):
            rn = A_normalized @ rn
            basis.append(rn)
        basis = torch.cat(basis, dim=-1)
        return basis

    def rnn(self, x):
        x = self.encoder(x)
        outputs = []
        for i in range(32):
            x = self.iterative_func(x)
            outputs.append(self.readout(x))
        y = torch.stack(outputs, dim=2)
        return y

    def _preprocess(self, tau: Dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs _"]:
        features = self._get_features(tau, theta)
        inputs = [features[k].reshape(-1, v.dim) for k, v in self.features.items()]
        x = torch.cat(inputs, dim=-1)
        return x

    def forward(self, tau: Dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
        x = self._preprocess(tau, theta)
        x0 = extract_param("x0", self.params_learn, theta).unsqueeze(-1)
        x_sol = tau["x_sol"].unsqueeze(-1)

        basis = self.get_krylov_basis(tau, theta)  # (bs, dim, iterations)
        coefficients = self.rnn(x)  # (bs, 1, iterations) coefficients of basis
        components = coefficients * basis  # (bs, dim, iterations)
        history = x0 + torch.cumsum(components, dim=-1)  # (bs, dim, iterations + 1)
        y = (history - x_sol).norm(dim=1)
        # y = history.transpose(1, 2)  # history_solution

        return y


def jvp(f, x: Tensor, v: Tensor, jvp_type: str, eps: float):
    if jvp_type == "forwardAD":
        # y, dvf = torch.func.jvp(f, (x,), (v,))
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
            except:
                f_true = torch.zeros_like(x)

            return f_true, None

        # forward gradient of f
        f_fwd = torch.sum(grad_y * d["dvf"], dim=1, keepdim=True) * v
        f_fwd = f_fwd.reshape(d["Nv"], bs, in_dim).mean(dim=0)  # average over Nv

        if d["grad_type"] == "f_fwd":
            return f_fwd, None

        # forward gradient of f_hat
        f_hat_fwd = torch.sum(grad_y * d["dvf_hat"], dim=1, keepdim=True) * v
        f_hat_fwd = f_hat_fwd.reshape(d["Nv"], bs, in_dim).mean(dim=0)  # average over Nv

        # full gradient of f_hat
        f_hat_true = grad(d["y_hat"], x, grad_outputs=grad_y, retain_graph=True)[0] / d["Nv"]

        if d["grad_type"] == "f_hat_true":
            return f_hat_true, None

        elif d["grad_type"] == "cv_fwd":
            # control variates
            # c = f_fwd.norm(dim=1, keepdim=True) / (f_hat_fwd - f_hat_true).norm(dim=1, keepdim=True)
            cv_fwd = f_fwd - (f_hat_fwd - f_hat_true)
            return cv_fwd, None

        else:
            raise NotImplementedError


class SurrogateSolverCNN2D(SurrogateSolver):
    def __init__(self, params_fix: Dict, params_learn: Dict, features: Dict, model: Dict) -> None:
        super().__init__(params_fix, params_learn, features)

        model.n_channels = len(features)
        self.model = instantiate(model)

    def forward(self, tau: Dict, theta: Tensor) -> Tensor:
        bs, n2 = tau["b"].shape
        features = self._get_features(tau, theta)
        n = int(n2**0.5)
        inputs = [features[k].reshape(-1, 1, n, n) for k, v in self.features.items()]
        x = torch.cat(inputs, dim=1)
        y = self.model(x)
        return y
