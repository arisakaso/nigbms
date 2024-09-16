from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

cs = ConfigStore.instance()


@dataclass
class TestFunctionConfig:
    """TestFunctionConfig class."""

    _target_: str = "nigbms.solvers.TestFunctionSolver"
    params_fix: DictConfig = DictConfig({})
    params_learn: DictConfig = DictConfig({})


cs.store(name="testfunction_solver_default", group="solver", node=TestFunctionConfig)


@dataclass
class PyTorchJacobiConfig:
    _target_: str = "nigbms.solvers.PyTorchJacobi"
    params_fix: DictConfig = DictConfig({"history_length": 100})
    params_learn: DictConfig = DictConfig({})


cs.store(name="pytorch_jacobi_default", group="solver", node=PyTorchJacobiConfig)


@dataclass
class PETScKSPConfig:
    _target_: str = "nigbms.solvers.PETScKSP"
    params_fix: DictConfig = DictConfig({"history_length": 100})
    params_learn: DictConfig = DictConfig({})
    debug: bool = True


cs.store(name="petsc_ksp_default", group="solver", node=PETScKSPConfig)


@dataclass
class PETScCGConfig(PETScKSPConfig):
    params_fix: DictConfig = DictConfig(
        {
            "history_length": 100,
            "ksp_type": "cg",
            "ksp_divtol": 1.0e10,
            "ksp_norm_type": "unpreconditioned",
            "ksp_initial_guess_nonzero": "true",
            "pc_type": "none",
        }
    )
    debug: bool = False


cs.store(name="petsc_cg_default", group="solver", node=PETScCGConfig)


@dataclass
class PETScJacobiConfig(PETScKSPConfig):
    params_fix: DictConfig = DictConfig(
        {
            "history_length": 100,
            "ksp_type": "richardson",
            "ksp_divtol": 1.0e10,
            "ksp_norm_type": "unpreconditioned",
            "ksp_initial_guess_nonzero": "true",
            "pc_type": "jacobi",
        }
    )
    debug: bool = False


cs.store(name="petsc_jacobi_default", group="solver", node=PETScJacobiConfig)
