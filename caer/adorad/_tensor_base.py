#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


class _TensorBase():
    def __init__(self):
        self.max_width = 2
        self.floating_dtype = None 
        self.sci_mode = False 
        self.int_mode = True  
        self.dtype = None 
        self.size = None 
        self.ndim = None 
        self.numel = None 


    # def numel(self):
    #     return self.numel 
    
    # def dim(self):
    #     return self.dim 

    # def size(self):
    #     return self.size  

    # def dtype(self):
    #     return self.dtype 