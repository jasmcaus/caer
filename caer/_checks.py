# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

def _check_size(size):
    """
    Common check to enforce type and sanity check on size tuples
    :param size: Should be a tuple of size 2 (width, height)
    :returns: True, or raises a ValueError
    """

    if not isinstance(size, (list, tuple)):
        raise ValueError("Size must be a tuple")
    if len(size) != 2:
        raise ValueError("Size must be a tuple of length 2")
    if size[0] < 0 or size[1] < 0:
        raise ValueError("Width and height must be >= 0")

    return True


def _check_mean_sub_values(value, channels):
    """
        Checks if mean subtraction values are valid based on the number of channels
        'value' must be a tuple of dimensions = number of channels
    Returns boolean:
        True -> Expression is valid
        False -> Expression is invalid
    """
    if value is None:
        raise ValueError('Value(s) specified is of NoneType()')
    
    if isinstance(value, tuple):
        # If not a tuple, we convert it to one
        try:
            value = tuple(value)
        except TypeError:
            value = tuple([value])
    
    if channels not in [1,3]:
        raise ValueError('Number of channels must be either 1 (Grayscale) or 3 (RGB/BGR)')

    if len(value) not in [1,3]:
        raise ValueError('Tuple length must be either 1 (subtraction over the entire image) or 3 (per channel subtraction)', value)
    
    if len(value) == channels:
        return True 

    else:
        raise ValueError(f'Expected a tuple of dimension {channels}', value) 