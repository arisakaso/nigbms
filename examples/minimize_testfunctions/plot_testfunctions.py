### PLOTTING ###
import numpy as np
from matplotlib import pyplot as plt


def plot_results(results, test_function, cfg):
    # draw contour lines
    _, ax = plt.subplots(1, 1, figsize=(8, 6))

    plot_params = {
        "f_true": {"color": "r", "alpha": 1.0},
        "f_fwd": {"color": "k", "alpha": 1.0},
        "f_hat_true": {"color": "g", "alpha": 1.0},
        "cv_fwd": {"color": "b", "alpha": 1.0},
    }
    labels = {
        "f_true": r"$\nabla f$",
        "f_fwd": r"$\mathbf{g}_\mathrm{v}$",
        "f_hat_true": r"$\nabla \hat f$",
        "cv_fwd": r"$\mathbf{h}_\mathrm{v}$",
    }
    for grad_type, steps_list in results.items():
        for i, steps in enumerate(steps_list):
            z = test_function(steps)
            ax.plot(
                np.arange(len(z)),
                z.mean(dim=1).ravel(),
                label=labels[grad_type] if i == 0 else None,
                **plot_params[grad_type],
            )
            if grad_type == "f_true":
                ax.set_ylim(z.mean(dim=1).min() * 0.1, z.mean(dim=1).max() * 10)

    ax.set_yscale("log")
    ax.legend()
    plt.savefig(
        f"{cfg.common.run_cfgs.function_name}{cfg.common.run_cfgs.dim}D.png", bbox_inches="tight", pad_inches=0.05
    )
