from nigbms.data.generate_clamped_beam3d import ClampedBeam3DParams, construct_petsc_clamped_beam3d
from nigbms.modules.tasks import PETScLinearSystemTask


# minimum test for construct_petsc_poisson2d_task
def test_construct_petsc_clamped_beam3d():
    parasm = ClampedBeam3DParams()
    task = construct_petsc_clamped_beam3d(parasm)
    assert isinstance(task, PETScLinearSystemTask)
