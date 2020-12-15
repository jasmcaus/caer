import torch
from torch.distributions import constraints
from torch.distributions.uniform import Uniform
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.utils import broadcast_all, euler_constant


def _moments(a, b, n):
    """
    Computes nth moment of Kumaraswamy using using torch.lgamma
    """
    arg1 = 1 + n / a
    log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b)
    return b * torch.exp(log_value)


class Kumaraswamy(TransformedDistribution):
    r"""
    Samples from a Kumaraswamy distribution.

    Example::

        >>> m = Kumaraswamy(torch.Tensor([1.0]), torch.Tensor([1.0]))
        >>> m.sample()  # sample from a Kumaraswamy distribution with concentration alpha=1 and beta=1
        tensor([ 0.1729])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """
    arg_constraints = {'concentration1': constraints.positive, 'concentration0': constraints.positive}
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, concentration1, concentration0, validate_args=None):
        self.concentration1, self.concentration0 = broadcast_all(concentration1, concentration0)
        finfo = torch.finfo(self.concentration0.dtype)
        base_dist = Uniform(torch.full_like(self.concentration0, 0),
                            torch.full_like(self.concentration0, 1))
        transforms = [PowerTransform(exponent=self.concentration0.reciprocal()),
                      AffineTransform(loc=1., scale=-1.),
                      PowerTransform(exponent=self.concentration1.reciprocal())]
        super(Kumaraswamy, self).__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Kumaraswamy, _instance)
        new.concentration1 = self.concentration1.expand(batch_shape)
        new.concentration0 = self.concentration0.expand(batch_shape)
        return super(Kumaraswamy, self).expand(batch_shape, _instance=new)

    @property
    def mean(self):
        return _moments(self.concentration1, self.concentration0, 1)

    @property
    def variance(self):
        return _moments(self.concentration1, self.concentration0, 2) - torch.pow(self.mean, 2)

    def entropy(self):
        t1 = (1 - self.concentration1.reciprocal())
        t0 = (1 - self.concentration0.reciprocal())
        H0 = torch.digamma(self.concentration0 + 1) + euler_constant
        return t0 + t1 * H0 - torch.log(self.concentration1) - torch.log(self.concentration0)
