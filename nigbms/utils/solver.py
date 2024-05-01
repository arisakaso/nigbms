import torch
from petsc4py import PETSc
from tensordict import TensorDict


def clear_petsc_options():
    opts = PETSc.Options()
    for key in opts.getAll().keys():
        opts.delValue(key)


def set_petsc_options(opts: dict):
    for key, value in opts.items():
        PETSc.Options().setValue(key, value)
    PETSc.Options().view()


def eyes_like(tensor: torch.Tensor):
    return torch.eye(tensor.shape[1], dtype=tensor.dtype, device=tensor.device).unsqueeze(0)


def rademacher_like(tensor: torch.Tensor):
    if isinstance(tensor, TensorDict):
        v = TensorDict({k: rademacher_like(v) for k, v in tensor.items()}, batch_size=tensor.batch_size)
    else:
        v = torch.randint(0, 2, tensor.shape, dtype=tensor.dtype, device=tensor.device) * 2 - 1
    return v


def bms(m, s):
    assert m.shape[: s.ndim] == s.shape
    target_shape = [1] * m.ndim
    target_shape[: s.ndim] = m.shape[: s.ndim]
    ms = m * s.reshape(target_shape)  # broadcast
    return ms


def initialize_parameters_fcn(m, scale=1e-3):
    if isinstance(m, torch.nn.Linear):
        print("initialized with scale", m, scale)
        torch.nn.init.uniform_(m.weight, -scale, scale)
        torch.nn.init.uniform_(m.bias, -scale, scale)
