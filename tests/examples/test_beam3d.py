from nigbms.tasks import PETScLinearSystemTask

from examples.beam3d.task import Beam3DParams, Beam3DTaskConstructor


class TestBeam3DTaskConstructor:
    def setup_method(self):
        self.constructor = Beam3DTaskConstructor()

    def test_construct_task(self) -> None:
        params = Beam3DParams()
        task = self.constructor(params)
        assert isinstance(task, PETScLinearSystemTask)
