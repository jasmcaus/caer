from .module import Module
from .. import functional as F


class ChannelShuffle(Module):
    r"""Divide the channels in a tensor of shape :math:`(*, C , H, W)`
    into g groups and rearrange them as :math:`(*, C \frac g, g, H, W)`,
    while keeping the original tensor shape.

    Args:
        groups (int): number of groups to divide channels in.

    Examples::

        >>> channel_shuffle = nn.ChannelShuffle(2)
        >>> input = torch.randn(1, 4, 2, 2)
        >>> print(input)
        [[[[1, 2],
           [3, 4]],
          [[5, 6],
           [7, 8]],
          [[9, 10],
           [11, 12]],
          [[13, 14],
           [15, 16]],
         ]]
        >>> output = channel_shuffle(input)
        >>> print(output)
        [[[[1, 2],
           [3, 4]],
          [[9, 10],
           [11, 12]],
          [[5, 6],
           [7, 8]],
          [[13, 14],
           [15, 16]],
         ]]
    """
    __constants__ = ['groups']

    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, input):
        return F.channel_shuffle(input, self.groups)

    def extra_repr(self):
        return 'groups={}'.format(self.groups)
