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

class Tensor(np.ndarray):
    def __repr__(self):
        return "<class 'caer.Tensor'>"

    def __new__(self, x, dtype=None):
        obj = np.asarray(x, dtype=dtype).view(Tensor)

        if len(obj.shape) > 1:
            self.size = (obj.shape[1], obj.shape[0])
        else:
            self.size = obj.shape
        return obj 
    
    # def __repr__(self):
    #     return "<class 'caer.Tensor'>"



def tensor(x, dtype=None):
    if not isinstance(x, (tuple, list, np.ndarray)):
        raise ValueError('Data needs to be (ideally) a list')

    return Tensor(x, dtype=dtype)
