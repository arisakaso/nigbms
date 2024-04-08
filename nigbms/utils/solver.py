import torch
from petsc4py import PETSc


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
