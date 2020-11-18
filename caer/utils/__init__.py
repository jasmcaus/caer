#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


from .validators import URLValidator
from .validators import validate_ipv6_address
from .validators import is_valid_url

from .dtype import (
    img_as_float32, 
    img_as_float64, 
    img_as_float,
    convert_to_float,
    img_as_int, 
    img_as_uint, 
    img_as_ubyte,
    img_as_bool, 
    dtype_limits
)

from .warnings import (
    all_warnings, 
    expected_warnings, 
    warn
)

# __all__ globals 
from .validators import __all__ as __all_validators__ 
from .dtype import __all__ as __all_dtype__

__all__ = __all_validators__ + __all_dtype__

del __all_dtype__
del __all_validators__