import torch
from nigbms.losses import SurrogateSolverLoss
from nigbms.models import MLPSkip
from nigbms.wrapper import NIGBMS, NIGBMSBundle
from torch.nn import Parameter


#### TEST FUNCTIONS ####
def sphere(x) -> torch.Tensor:
    """Sphere function."""
    return torch.sum(x**2, dim=-1, keepdim=True)


def rosenbrock(x) -> torch.Tensor:
    """Rosenbrock function."""
    x1 = x[..., :-1]
    x2 = x[..., 1:]
    return torch.sum(100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2, dim=-1, keepdim=True)


def rosenbrock_separate(x) -> torch.Tensor:
    """Rosenbrock function (easy)."""
    assert x.shape[-1] % 2 == 0, "Dimension must be even."
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.sum((1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2, dim=-1, keepdim=True)


def rastrigin(x) -> torch.Tensor:
    """Rastrigin function."""
    A = 10
    n = x.shape[-1]
    return A * n + torch.sum(x**2 - A * torch.cos(x * torch.pi * 2), dim=-1, keepdim=True)


if __name__ == "__main__":
    # set up
    torch.set_default_device("cuda")
    test_func = sphere
    bs = 100
    dim = 128
    theta = Parameter(torch.randn(bs, dim))
    surrogate = MLPSkip(
        in_dim=dim,
        out_dim=1,
        num_layers=2,
        n_units=512,
        hidden_activation=torch.nn.GELU(),
    )
    opt_outer = torch.optim.SGD([theta], lr=1e-2)
    opt_inner = torch.optim.SGD(surrogate.parameters(), lr=1e-3)
    loss_outer = torch.sum
    loss_inner = SurrogateSolverLoss(weights={"dvf_loss": 1}, reduce=True)

    # minimize the test function by updating theta
    for i in range(500):
        # clear gradients
        opt_outer.zero_grad()

        # forward pass
        bundle = NIGBMSBundle(  # package all the necessary information
            f=test_func,  # f and f_hat MUST have the same input and output shape.
            f_hat=surrogate,
            loss=loss_inner,
            opt=opt_inner,
            grad_type="cv_fwd",
            jvp_type="forwardAD",
            eps=0,
            Nv=1,
            v_dist="rademacher",
            v_scale=1.0,
            additional_steps=1,
        )
        y = NIGBMS.apply(theta, bundle)  # This line does the magic. See nigbms/wrapper.py for details.

        # backprop as usual
        loss_outer(y).backward()

        # logging and monitoring
        ref = theta.clone()  # copy to get the true gradient to compare
        f_true = torch.autograd.grad(test_func(ref).sum(), ref)[0]
        sim = torch.cosine_similarity(f_true, theta.grad, dim=1)
        if i % 100 == 0:
            print(f"{i=}, {y.mean()=:.3g}, {y.max()=:.3g}, {y.min()=:.3g}, {sim.mean()=:.3g}")

        # update
        opt_outer.step()

    print("Final Result:")
    print(f"{i=}, {y.mean()=:.3g}, {y.max()=:.3g}, {y.min()=:.3g}, {sim.mean()=:.3g}")
