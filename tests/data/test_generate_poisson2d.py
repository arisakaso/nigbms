from nigbms.data.generate_poisson2d import Poisson2DParams, construct_petsc_poisson2d_task


# minimum test for construct_petsc_poisson2d_task
def test_construct_petsc_poisson2d_task():
    parasm = Poisson2DParams()
    task = construct_petsc_poisson2d_task(parasm)
    assert task is not None
