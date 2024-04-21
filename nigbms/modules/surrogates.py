import torch
from hydra.utils import instantiate
from torch import Tensor

from nigbms.modules.solvers import _Solver


class _SurrogateSolver(_Solver):
    def __init__(self, params_fix: dict, params_learn: dict, features: dict) -> None:
        super().__init__(params_fix, params_learn)
        self.features = features

    def _make_features(self, tau: dict, theta: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, tau: dict, theta: Tensor) -> Tensor:
        raise NotImplementedError


class SurrogateSolverMLP(_SurrogateSolver):
    def __init__(self, params_fix: dict, params_learn: dict, features: dict, model: dict) -> None:
        super().__init__(params_fix, params_learn, features)

        in_dim = 0
        for v in features.values():
            in_dim += v["dim"]
        model.layers[0] = in_dim
        self.model = instantiate(model)

    def forward(self, tau: dict, theta: Tensor) -> Tensor:
        features = self._get_features(tau, theta)
        x = torch.cat([features[k] for k in self.features.keys()], dim=1)
        y = self.model(x)
        return y


class SurrogateSolverCNN1D(_SurrogateSolver):
    def __init__(self, params_fix: dict, params_learn: dict, features: dict, model: dict) -> None:
        super().__init__(params_fix, params_learn, features)

        model.in_channels = len(features)
        self.model = instantiate(model)

    def _preprocess(self, tau: dict, theta: Tensor) -> Tensor:
        features = self._get_features(tau, theta)
        inputs = [features[k].reshape(-1, 1, v.dim) for k, v in self.features.items()]
        x = torch.cat(inputs, dim=1)
        return x

    def forward(self, tau: dict, theta: Tensor) -> Tensor:
        x = self._preprocess(tau, theta)
        y = self.model(x)
        return y


class SurrogateSolverCNN2D(_SurrogateSolver):
    def __init__(self, params_fix: dict, params_learn: dict, features: dict, model: dict) -> None:
        super().__init__(params_fix, params_learn, features)

        model.n_channels = len(features)
        self.model = instantiate(model)

    def forward(self, tau: dict, theta: Tensor) -> Tensor:
        bs, n2 = tau["b"].shape
        features = self._get_features(tau, theta)
        n = int(n2**0.5)
        inputs = [features[k].reshape(-1, 1, n, n) for k, v in self.features.items()]
        x = torch.cat(inputs, dim=1)
        y = self.model(x)
        return y


# class SurrogateSolverIterative(_SurrogateSolver):
#     def __init__(
#         self,
#         params_fix: dict,
#         params_learn: dict,
#         features: dict,
#         encoder: dict,
#         iterative_func: dict,
#         iterations: int,
#     ) -> None:
#         super().__init__(params_fix, params_learn, features)
#         in_dim = 0
#         for v in features.values():
#             in_dim += v["dim"]
#         encoder.layers[0] = in_dim
#         iterative_func.layers[0] = encoder.layers[-1] + 1  # add index
#         iterative_func.layers[-1] = encoder.layers[-1]
#         self.encoder = instantiate(encoder)
#         self.iterative_func = instantiate(iterative_func)
#         self.readout = torch.nn.Linear(iterative_func.layers[-1] + 1, 1)
#         self.iterations = iterations

#     def _preprocess(self, tau: dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs _"]:
#         features = self._get_features(tau, theta)
#         inputs = [features[k].reshape(-1, v.dim) for k, v in self.features.items()]
#         x = torch.cat(inputs, dim=-1)
#         return x, features

#     def forward(self, tau: dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
#         x, features = self._preprocess(tau, theta)
#         ref = features["e0"].norm(dim=1, keepdim=True)
#         enc = self.encoder(x)
#         idx = torch.linspace(0, 1, self.iterations + 1, device=x.device).repeat(x.shape[0], 1)  # step index
#         enc = torch.cat([enc, idx[:, [0]]], dim=-1)
#         outputs = []
#         for i in range(self.iterations):
#             enc = self.iterative_func(enc)
#             enc = torch.cat([enc, idx[:, [i + 1]]], dim=-1)
#             outputs.append(self.readout(enc) + ref)
#         y = torch.stack(outputs, dim=1)
#         return y


# class SurrogateSolverKrylov(_SurrogateSolver):
#     def __init__(
#         self,
#         params_fix: dict,
#         params_learn: dict,
#         features: dict,
#         model: dict,
#         scale=0.9,
#     ) -> None:
#         super().__init__(params_fix, params_learn, features)

#         in_dim = 0
#         for v in features.values():
#             in_dim += v["dim"]
#         model.layers[0] = in_dim
#         self.model = instantiate(model)
#         self.scale = scale

#     def get_krylov_basis(self, tau, theta):
#         b = tau["features"]["b"].unsqueeze(-1)
#         A = tau["A"]
#         x0 = extract_param("x0", self.params_learn, theta).unsqueeze(-1)
#         r0 = b - A @ x0
#         rn = r0
#         basis = [rn]
#         A_normalized = A / A.norm(dim=(1, 2), keepdim=True, p=2)
#         for i in range(A.shape[1] - 1):
#             # rn = normalize(A @ rn) * self.scale**i
#             rn = A_normalized @ rn
#             basis.append(rn)
#         basis = torch.cat(basis, dim=-1)
#         return basis

#     def _preprocess(self, tau: dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs _"]:
#         features = self._get_features(tau, theta)
#         inputs = [features[k].reshape(-1, v.dim) for k, v in self.features.items()]
#         x = torch.cat(inputs, dim=-1)
#         return x

#     def forward(self, tau: dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
#         x = self._preprocess(tau, theta)
#         x0 = extract_param("x0", self.params_learn, theta).unsqueeze(-1)
#         x_sol = tau["x_sol"].unsqueeze(-1)

#         basis = self.get_krylov_basis(tau, theta)  # (bs, dim, iterations)
#         coefficients = self.model(x).unsqueeze(1)  # (bs, 1, iterations) coefficients of basis
#         components = coefficients * basis  # (bs, dim, iterations)
#         history = x0 + torch.cumsum(components, dim=-1)  # (bs, dim, iterations + 1)
#         y = (history - x_sol).norm(dim=1)
#         # y = history.transpose(1, 2)  # history_solution

#         return y


# class SurrogateSolverKrylovSeq(_SurrogateSolver):
#     def __init__(
#         self,
#         params_fix: dict,
#         params_learn: dict,
#         features: dict,
#         encoder: dict,
#         iterative_func,
#     ) -> None:
#         super().__init__(params_fix, params_learn, features)

#         in_dim = 0
#         for v in features.values():
#             in_dim += v["dim"]
#         encoder.layers[0] = in_dim
#         self.encoder = instantiate(encoder)
#         self.iterative_func = instantiate(iterative_func)
#         self.readout = torch.nn.Linear(iterative_func.layers[-1], 1)

#     def get_krylov_basis(self, tau, theta):
#         b = tau["features"]["b"].unsqueeze(-1)
#         A = tau["A"]
#         x0 = extract_param("x0", self.params_learn, theta).unsqueeze(-1)
#         r0 = b - A @ x0
#         rn = normalize(r0)
#         basis = [rn]
#         A_normalized = A / A.norm(dim=(1, 2), keepdim=True, p=2)
#         for i in range(A.shape[1] - 1):
#             rn = A_normalized @ rn
#             basis.append(rn)
#         basis = torch.cat(basis, dim=-1)
#         return basis

#     def rnn(self, x):
#         x = self.encoder(x)
#         outputs = []
#         for i in range(32):
#             x = self.iterative_func(x)
#             outputs.append(self.readout(x))
#         y = torch.stack(outputs, dim=2)
#         return y

#     def _preprocess(self, tau: dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs _"]:
#         features = self._get_features(tau, theta)
#         inputs = [features[k].reshape(-1, v.dim) for k, v in self.features.items()]
#         x = torch.cat(inputs, dim=-1)
#         return x

#     def forward(self, tau: dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
#         x = self._preprocess(tau, theta)
#         x0 = extract_param("x0", self.params_learn, theta).unsqueeze(-1)
#         x_sol = tau["x_sol"].unsqueeze(-1)

#         basis = self.get_krylov_basis(tau, theta)  # (bs, dim, iterations)
#         coefficients = self.rnn(x)  # (bs, 1, iterations) coefficients of basis
#         components = coefficients * basis  # (bs, dim, iterations)
#         history = x0 + torch.cumsum(components, dim=-1)  # (bs, dim, iterations + 1)
#         y = (history - x_sol).norm(dim=1)
#         # y = history.transpose(1, 2)  # history_solution

#         return y


# class SurrogateSolverHyper(_SurrogateSolver):
#     def __init__(
#         self,
#         params_fix: dict,
#         params_learn: dict,
#         features: dict,
#         mnet: dict,
#         hnet: dict,
#     ) -> None:
#         super().__init__(params_fix, params_learn, features)
#         mnet = instantiate(mnet)
#         m_func, m_params = functorch.make_functional(mnet)
#         self.mp_shapes = [mp.shape for mp in m_params]
#         self.mp_offsets = [0] + list(np.cumsum([mp.numel() for mp in m_params]))
#         hnet.layers[-1] = int(self.mp_offsets[-1])
#         hnet = instantiate(hnet)
#         self.feature_extractor = torch.nn.Sequential(
#             torch.nn.Linear(128, 128),
#             torch.nn.SiLU(),
#         )
#         self.m_func = torch.vmap(m_func)
#         self.hnet = hnet

#     def generate_params(self, tau, theta):
#         features = self._get_features(tau, theta)
#         z = torch.cat([features["x_sol"], features["b"], features["x0"].detach().clone()], dim=1)
#         params = self.hnet(z)
#         self.params_lst = []
#         for i, shape in enumerate(self.mp_shapes):
#             j0, j1 = self.mp_offsets[i], self.mp_offsets[i + 1]
#             self.params_lst.append(params[..., j0:j1].reshape(-1, *shape))

#     def forward(self, tau: dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
#         features = self._get_features(tau, theta)
#         x = features["x0"]

#         # x = self.feature_extractor(x)
#         y = self.m_func(self.params_lst, x)
#         return y


# class SurrogateSolverOperator(_SurrogateSolver):
#     def __init__(
#         self,
#         params_fix: dict,
#         params_learn: dict,
#         features: dict,
#         branch: dict,
#         trunk: dict,
#     ) -> None:
#         super().__init__(params_fix, params_learn, features)

#         self.branch = instantiate(branch)
#         self.trunk = instantiate(trunk)

#     def forward(self, tau: dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
#         features = self._get_features(tau, theta)
#         x = features["x0_freq"]
#         z = torch.cat([features["x_sol_freq"], features["b_freq"]], dim=1)
#         b = self.branch(z)
#         t = self.trunk(x)
#         y = torch.sum(b * t, dim=1, keepdim=True)
#         return y

# class SurrogateSolverTransformer(_SurrogateSolver):
#     def __init__(self, params_fix: dict, params_learn: dict, features: dict, model: dict) -> None:
#         super().__init__(params_fix, params_learn, features)
#         self.model = torch.nn.Sequential(
#             torch.nn.TransformerEncoderLayer(
#                 d_model=4,
#                 nhead=4,
#                 dim_feedforward=128,
#                 dropout=0,
#                 activation="gelu",
#                 batch_first=True,
#             ),
#             torch.nn.TransformerEncoderLayer(
#                 d_model=4,
#                 nhead=4,
#                 dim_feedforward=128,
#                 dropout=0,
#                 activation="gelu",
#                 batch_first=True,
#             ),
#             torch.nn.Flatten(),
#             torch.nn.Linear(32 * 4, 1),
#         )

#     def _preprocess(self, tau: dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs _"]:
#         features = self._get_features(tau, theta)
#         inputs = [features[k].reshape(-1, v.dim, 1) for k, v in self.features.items()]
#         x = torch.cat(inputs, dim=-1)  # need (bs, length, features)
#         return x

#     def forward(self, tau: dict, theta: Float[Tensor, "bs _"]) -> Float[Tensor, "bs"]:
#         x = self._preprocess(tau, theta)
#         y = self.model(x)
#         return y
