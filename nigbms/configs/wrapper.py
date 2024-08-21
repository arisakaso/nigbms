from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

cs = ConfigStore.instance()


@dataclass
class WrappedSolverConfig:
    """WrappedSolverConfig class.
    Solver, Surrogate, and Constructor are not included and should be specified later"""

    _target_: str = "nigbms.modules.wrapper.WrappedSolver"
    _recursive_: bool = False
    hparams: DictConfig = DictConfig(
        {
            "opt": {
                "_target_": "torch.optim.SGD",
                "lr": 0.001,
            },
            "loss": {
                "_target_": "nigbms.modules.losses.SurrogateSolverLoss",
                "weights": {"dvf_loss": 1},
                "reduce": True,
            },
            "clip": 100.0,
            "grad_type": "cv_fwd",
            "jvp_type": "forwardFD",
            "eps": 1.0e-10,
            "Nv": 1,
            "v_scale": 1.0,
            "v_dist": "rademacher",
        }
    )


cs.store(name="wrapped_solver_default", group="wrapper", node=WrappedSolverConfig)
