#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


from .extract_frames import (
    extract_frames,
    __all__ as __all_extract__
)

from .livestream import (
    LiveStream,
    __all__ as __all_live__ 
)

from .stream import (
    Stream,
    __all__ as __all_str__
)

from .gpufilestream import (
    GPUFileStream,
    __all__ as __all_gpu__ 
)

from .frames_and_fps import (
    count_frames,
    get_fps,
    __all__ as __all_ffps__
)


__all__ = __all_extract__ + __all_str__ + __all_live__ + __all_ffps__ + __all_gpu__


# Stop polluting the namespace
del __all_extract__
del __all_str__
del __all_ffps__
del __all_gpu__
del __all_live__