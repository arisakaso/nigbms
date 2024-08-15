from nigbms.data.generate_poisson2d import Poisson2DParams, construct_petsc_poisson2d_task
from nigbms.modules.tasks import PETScLinearSystemTask


# minimum test for construct_petsc_poisson2d_task
def test_construct_petsc_poisson2d_task() -> None:
    parasm = Poisson2DParams()
    task = construct_petsc_poisson2d_task(parasm)
    assert isinstance(task, PETScLinearSystemTask)
