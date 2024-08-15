# %%

import numpy as np
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


class Distribution(object):
    def __init__(self, shape):
        self.shape = shape

    def sample(self, seed=None) -> np.ndarray:
        raise NotImplementedError


class NumpyLogUniform(Distribution):
    def __init__(self, shape, lb, ub):
        super().__init__(shape)
        self.log_lb = np.log(lb)
        self.log_ub = np.log(ub)

    def sample(self, seed=None) -> np.ndarray:
        np.random.seed(seed)
        return np.exp(np.random.uniform(self.log_lb, self.log_ub, self.shape))


class NumpyConstant(Distribution):
    def __init__(self, shape, value):
        super().__init__(shape)
        if shape is not None:
            self.const = np.full(shape, value)
        else:
            self.const = value

    def sample(self, seed=None) -> np.ndarray:
        return self.const


class NumpyUniform(Distribution):
    def __init__(self, shape, lb, ub):
        super().__init__(shape)
        self.lb = lb
        self.ub = ub

    def sample(self, seed=None) -> np.ndarray:
        np.random.seed(seed)
        return np.random.uniform(self.lb, self.ub, self.shape)


class NumpyNormal(Distribution):
    def __init__(self, shape, mean, std):
        super().__init__(shape)
        self.mean = mean
        self.std = std

    def sample(self, seed=None) -> np.ndarray:
        np.random.seed(seed)
        return np.random.normal(self.mean, self.std, self.shape)
