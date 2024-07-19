# %%
from typing import Any, Dict, List

import numpy as np
import scipy.sparse as sp
import torch
import torch.distributions as dist
from petsc4py import PETSc


def petsc_collate_fn(batch: List[Any]) -> Any:
    A, b, x, rtol, maxiter, features = zip(*batch)
    features = torch.stack(features)
    tau = {
        "A": A,
        "b": b,
        "x_sol": x,
        "rtol": rtol,
        "maxiter": maxiter,
        "features": features,
    }
    return tau


def scipycsr2petscmat(csr_matrix: sp.csr_matrix) -> PETSc.Mat:
    """Converts a scipy csr matrix to a petsc matrix

    Args:
        csr_matrix (sp.csr_matrix): scipy csr matrix

    Returns:
        PETSc.Mat: petsc matrix
    """

    nrows, ncols = csr_matrix.shape
    nnz = csr_matrix.nnz
    row_ptr, col_idx, values = csr_matrix.indptr, csr_matrix.indices, csr_matrix.data

    # Create the PETSc matrix using the createAIJ method
    petsc_matrix = PETSc.Mat().createAIJ(
        size=(nrows, ncols),
        nnz=nnz,
        csr=(row_ptr, col_idx, values),
        comm=PETSc.COMM_WORLD,
    )

    # Assemble the PETSc matrix
    petsc_matrix.assemblyBegin()
    petsc_matrix.assemblyEnd()

    return petsc_matrix


def torchcoo2scipycsr(torch_coo):
    # Sparse Tensorを統合（coalesce）
    if not torch_coo.is_coalesced():
        torch_coo = torch_coo.coalesce()
    torch_coo = torch_coo.detach().cpu()
    # PyTorchのTensorからデータを取得
    rows = torch_coo.indices()[0].numpy()
    cols = torch_coo.indices()[1].numpy()
    data = torch_coo.values().numpy()

    # SciPyのCSR形式のスパース行列を作成
    csr_matrix = sp.csr_matrix((data, (rows, cols)), shape=torch_coo.shape)

    return csr_matrix


def torchcoo2petscmat(torch_coo: torch.sparse.FloatTensor) -> PETSc.Mat:
    """Converts a torch sparse tensor to a petsc matrix

    Args:
        torch_coo (torch.sparse.FloatTensor): torch sparse tensor

    Returns:
        PETSc.Mat: petsc matrix
    """

    # torch coo to scipy csr
    csr_matrix = torchcoo2scipycsr(torch_coo)
    petsc_matrix = scipycsr2petscmat(csr_matrix)

    return petsc_matrix


def scipycoo2torchcoo(scipy_coo):
    # SciPyのCOO行列からデータを取得
    values = torch.from_numpy(scipy_coo.data)
    indices = torch.from_numpy(
        np.vstack((scipy_coo.row, scipy_coo.col)).astype(np.int64)
    )

    # PyTorchのCOO形式のSparse Tensorを作成
    torch_coo = torch.sparse_coo_tensor(indices, values, scipy_coo.shape)

    return torch_coo


def petscmat2scipycsr(petsc_matrix: PETSc.Mat) -> sp.csr_matrix:
    """Converts a petsc matrix to a scipy csr matrix

    Args:
        petsc_matrix (PETSc.Mat): petsc matrix

    Returns:
        sp.csr_matrix: scipy csr matrix
    """
    row_ptr, col_idx, values = petsc_matrix.getValuesCSR()
    return sp.csr_matrix((values, col_idx, row_ptr))


def numpy2petscvec(np_array: np.ndarray) -> PETSc.Vec:
    """Converts a numpy array to a petsc vector

    Args:
        np_array (np.ndarray): numpy array

    Returns:
        PETSc.Vec: petsc vector
    """
    petsc_vec = PETSc.Vec().createWithArray(np_array, comm=PETSc.COMM_WORLD)
    return petsc_vec


def tensor2petscvec(torch_tensor: torch.Tensor) -> PETSc.Vec:
    """Converts a torch tensor to a petsc vector

    Args:
        torch_tensor (torch.Tensor): torch tensor

    Returns:
        PETSc.Vec: petsc vector
    """
    np_array = torch_tensor.detach().cpu().numpy()
    petsc_vec = numpy2petscvec(np_array)
    return petsc_vec


def scipycsr2torchcsr(csr_matrix: sp.csr_matrix) -> torch.Tensor:
    """Converts a scipy csr matrix to a torch sparse tensor

    Args:
        csr_matrix (sp.csr_matrix): scipy csr matrix

    Returns:
        torch.sparse.FloatTensor: torch sparse tensor
    """

    return torch.sparse_csr_tensor(
        csr_matrix.indptr, csr_matrix.indices, csr_matrix.data, size=csr_matrix.shape
    )


# %%
def relative_mse_loss(y, y_hat, eps=1e-16):
    return torch.mean(torch.square(y - y_hat) / torch.square(y))


class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        lb = torch.tensor(lb)
        ub = torch.tensor(ub)
        super().__init__(dist.Uniform(lb.log(), ub.log()), dist.ExpTransform())


class Constant(dist.TransformedDistribution):
    def __init__(self, val):
        super().__init__(
            dist.Bernoulli(probs=0), dist.AffineTransform(loc=val, scale=0)
        )


# %%
def set_my_style():
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_context("paper", font_scale=1.5)
    plt.rcParams["figure.figsize"] = (8, 6)  # figure size in inch, 横×縦
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams["font.family"] = "Times New Roman"  # 全体のフォントを設定
    plt.rcParams["xtick.direction"] = "in"  # x axis in
    plt.rcParams["ytick.direction"] = "in"  # y axis in
    plt.rcParams["axes.linewidth"] = 1.0  # axis line width
    plt.rcParams["axes.grid"] = True  # make grid
    # plt.rcParams["font.sans-serif"] = "cm"
    # plt.rcParams["mathtext.fontset"] = "cm"
    # plt.rcParams["font.cm"] = "Computer Modern"

    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.default"] = "it"
    plt.rcParams["mathtext.it"] = "cmmi10"
    plt.rcParams["mathtext.bf"] = "CMU serif:italic:bold"
    plt.rcParams["mathtext.rm"] = "cmb10"
    plt.rcParams["mathtext.fallback"] = "cm"


def extract_param(param: str, params_learn: Dict, theta: torch.Tensor) -> torch.Tensor:
    v = params_learn[param]
    return theta[..., v.start : v.end]


def flatten_theta(theta: Dict[str, torch.Tensor], opts2learn: Dict) -> torch.Tensor:
    """flatten theta to

    Args:
        theta (Dict[str, torch.Tensor]): key: PETSc option name
        opts2learn (Dict): key: PETSc option name

    Returns:
        theta_flat (torch.Tensor): (batch_size, num_params)
    """

    theta_flat = torch.hstack([theta[k] for k in opts2learn])

    return theta_flat


def unflatten_theta(
    theta_flat: torch.Tensor, opts2learn: Dict
) -> Dict[str, torch.Tensor]:
    """unflatten theta

    Args:
        theta_flat (torch.Tensor): flattened theta (batch_size, num_params)
        opts2learn (Dict): key: PETSc option name

    Returns:
        theta (Dict[str, torch.Tensor]): key: PETSc option name, value: tensor of shape (batch_size, option_dim)
    """

    theta = {}
    idx = 0
    for k in opts2learn:
        dim = opts2learn[k]["dim"]
        theta[k] = theta_flat[..., idx : idx + dim]
        idx += dim

    return theta


def get_opts2learn(theta: Dict):
    opts2learn = {}
    for k, v in theta.items():
        opts2learn[k] = {"dim": v.shape[-1]}
    return opts2learn


def get_features(tau: Dict):
    features = {}
    for k, v in tau["features"].items():
        features[k] = {"dim": v.shape[1]}
    return features


# %%
