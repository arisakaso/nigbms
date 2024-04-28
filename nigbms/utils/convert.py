import numpy as np
import scipy.sparse as sp
import torch
from petsc4py import PETSc
from tensordict import TensorDict


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
    indices = torch.from_numpy(np.vstack((scipy_coo.row, scipy_coo.col)).astype(np.int64))

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

    return torch.sparse_csr_tensor(csr_matrix.indptr, csr_matrix.indices, csr_matrix.data, size=csr_matrix.shape)


def tensordict2list(t: TensorDict):
    return zip(*[(k, v) for k, v in t.items()], strict=False)


def list2tensordict(ks, vs):
    return TensorDict({k: v for k, v in zip(ks, vs, strict=False)})
