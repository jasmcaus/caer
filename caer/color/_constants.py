#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-21 The Caer Authors <http://github.com/jasmcaus>


## Color Spaces
IMREAD_COLOR = 1
BGR2RGB = 4  
BGR2GRAY = 6  
RGB2GRAY = 7
GRAY2BGR = 8
GRAY2RGB = 8
BGR2HSV = 40 
RGB2BGR = BGR2RGB 
RGB2HSV = 41 
BGR2LAB = 44  
RGB2LAB = 45 
BGR2HLS = 52
RGB2HLS = 53
HSV2BGR = 54
HSV2RGB = 55
LAB2BGR = 56
LAB2RGB = 57
HLS2BGR = 60
HLS2RGB = 61



__all__ = [d for d in dir() if not d.startswith('_')]