import torch
from pytorch_lightning.callbacks import Callback

from nigbms.solvers import _PytorchIterativeSolver


class CosineSimilarityCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        assert isinstance(
            pl_module.wrapped_solver.solver, _PytorchIterativeSolver
        ), "Only PytorchIterativeSolver is supported"

        tau = batch
        theta = outputs["theta"]

        theta_ref = theta.clone()  # copy to get the true gradient
        y_ref = pl_module.wrapped_solver(tau, theta_ref, mode="test")
        loss_ref = pl_module.loss(tau, theta_ref, y_ref)["loss"]

        # Calculate true gradient
        f_true = torch.autograd.grad(loss_ref, theta_ref)[0]

        # Cosine similarity
        sim = torch.cosine_similarity(f_true, theta.grad, dim=1)

        # Log similarity
        trainer.logger.log_metrics({"surrogate/sim": sim.mean()}, step=trainer.global_step)
