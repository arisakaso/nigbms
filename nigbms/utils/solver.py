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


def add_tensordicts(tensordict1, tensordict2):
    """
    2つのtensordictを足し合わせる関数

    Args:
        tensordict1 (TensorDict): 1つ目のtensordict
        tensordict2 (TensorDict): 2つ目のtensordict

    Returns:
        TensorDict: 2つのtensordictを足し合わせた結果
    """
    # 2つのtensordictのキーが同じであることを確認
    assert set(tensordict1.keys()) == set(tensordict2.keys()), "Keys in the two tensordicts must be the same."

    # 各keyに対応するtensorを足し合わせる
    result = {k: tensordict1[k] + tensordict2[k] for k in tensordict1.keys()}

    # 新しいtensordictを作成して返す
    return TensorDict(result, batch_size=tensordict1.batch_size)
