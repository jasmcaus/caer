#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


from .extract_frames import extract_frames

# from .livestream import LiveStream

from .stream import Stream

from .gpufilestream import GPUFileStream

from .frames_and_fps import count_frames
from .frames_and_fps import get_fps


# __all__ globals 
from .extract_frames import __all__ as __all_extract__ 
# from .livestream import __all__ as __all_livevs__
from .gpufilestream import __all__ as __all_gpufilevs__
from .stream import __all__ as __all_vs__
from .frames_and_fps import __all__ as __all_ffps__


__all__ = __all_extract__ + __all_vs__ + __all_ffps__ + __all_gpufilevs__
# __all__ = __all_extract__  + __all_livevs__ + __all_vs__ + __all_ffps__ + __all_gpufilevs__

# Stop polluting the namespace
del __all_extract__
del __all_ffps__
del __all_gpufilevs__
# del __all_livevs__
del __all_vs__