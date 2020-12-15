import torch
from torch._six import inf
from torch.distributions.distribution import Distribution
from torch.distributions import Categorical
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all


class Multinomial(Distribution):
    r"""
    Creates a Multinomial distribution parameterized by :attr:`total_count` and
    either :attr:`probs` or :attr:`logits` (but not both). The innermost dimension of
    :attr:`probs` indexes over categories. All other dimensions index over batches.

    Note that :attr:`total_count` need not be specified if only :meth:`log_prob` is
    called (see example below)

    .. note:: :attr:`probs` must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1.

    -   :meth:`sample` requires a single shared `total_count` for all
        parameters and samples.
    -   :meth:`log_prob` allows different `total_count` for each parameter and
        sample.

    Example::

        >>> m = Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))
        >>> x = m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 21.,  24.,  30.,  25.])

        >>> Multinomial(probs=torch.tensor([1., 1., 1., 1.])).log_prob(x)
        tensor([-4.1338])

    Args:
        total_count (int): number of trials
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities
    """
    arg_constraints = {'probs': constraints.simplex,
                       'logits': constraints.real}
    total_count: int

    @property
    def mean(self):
        return self.probs * self.total_count

    @property
    def variance(self):
        return self.total_count * self.probs * (1 - self.probs)

    def __init__(self, total_count=1, probs=None, logits=None, validate_args=None):
        if not isinstance(total_count, int):
            raise NotImplementedError('inhomogeneous total_count is not supported')
        self.total_count = total_count
        self._categorical = Categorical(probs=probs, logits=logits)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super(Multinomial, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Multinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count
        new._categorical = self._categorical.expand(batch_shape)
        super(Multinomial, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @constraints.dependent_property
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    @property
    def logits(self):
        return self._categorical.logits

    @property
    def probs(self):
        return self._categorical.probs

    @property
    def param_shape(self):
        return self._categorical.param_shape

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        samples = self._categorical.sample(torch.Size((self.total_count,)) + sample_shape)
        # samples.shape is (total_count, sample_shape, batch_shape), need to change it to
        # (sample_shape, batch_shape, total_count)
        shifted_idx = list(range(samples.dim()))
        shifted_idx.append(shifted_idx.pop(0))
        samples = samples.permute(*shifted_idx)
        counts = samples.new(self._extended_shape(sample_shape)).zero_()
        counts.scatter_add_(-1, samples, torch.ones_like(samples))
        return counts.type_as(self.probs)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        logits = logits.clone(memory_format=torch.contiguous_format)
        log_factorial_n = torch.lgamma(value.sum(-1) + 1)
        log_factorial_xs = torch.lgamma(value + 1).sum(-1)
        logits[(value == 0) & (logits == -inf)] = 0
        log_powers = (logits * value).sum(-1)
        return log_factorial_n - log_factorial_xs + log_powers
