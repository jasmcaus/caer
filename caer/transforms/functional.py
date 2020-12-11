#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>

import numpy as np 
Tensor = np.ndarray 


def _get_img_size(img) -> Tensor:
    r"""
        Returns image size as (width, height)
    """
    h, w = img.shape[:2]

    return (w, h)


def _get_num_channels(img) -> int:
    r"""
        We assume only images of 1 and 3 channels
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        return 3
    
    else:
        return 1
    

def _is_numpy_array(img) -> bool:
    return isinstance(img, Tensor)


def _is_numpy_image(img) -> bool:
    return img.ndim in {2, 3}


def is_tuple(x):
    return isinstance(x, tuple)


def is_list(x):
    return isinstance(x, list)


def is_numeric(x):
    return isinstance(x, int)


def is_numeric_list_or_tuple(x):
    for i in x:
        if not is_numeric(i):
            return False
    return True