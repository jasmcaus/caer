#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


#pylint:disable=undefined-all-variable

from .paths import (
    list_media,
    list_images,
    list_videos,
    listdir,
    is_image,
    is_video,
    cwd,
    exists,
    get_size,
    abspath,
    isdir,
    isfile,
    mkdir,
    osname,
    chdir,
    minijoin,
    dirname,

    # Variables
    _acceptable_video_formats,
    _acceptable_image_formats,

    __all__ as __all_paths__,
)

__all__ = __all_paths__ 

# Stop polluting the namespace
del __all_paths__