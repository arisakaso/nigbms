# %%

import torch
import torch.distributions as dist


class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        lb = torch.tensor(lb)
        ub = torch.tensor(ub)
        super().__init__(dist.Uniform(lb.log(), ub.log()), dist.ExpTransform())


class Constant(dist.TransformedDistribution):
    def __init__(self, val):
        super().__init__(dist.Bernoulli(probs=0), dist.AffineTransform(loc=val, scale=0))
