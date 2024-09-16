from nigbms.tasks import PETScLinearSystemTask

from examples.beam3d.task import ClampedBeam3DParams, construct_petsc_beam3d


# minimum test for construct_petsc_poisson2d_task
def test_construct_petsc_clamped_beam3d():
    parasm = ClampedBeam3DParams()
    task = construct_petsc_beam3d(parasm)
    assert isinstance(task, PETScLinearSystemTask)
