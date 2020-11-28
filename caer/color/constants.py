#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


## Color Spaces
IMREAD_COLOR = 1 # cv.IMREAD_COLOR
BGR2RGB = 4  # cv.COLOR_BGR2RGB
BGR2GRAY = 6  # cv.COLOR_BGR2GRAY
RGB2GRAY = 7  # cv.COLOR_RGB2GRAY
BGR2HSV = 40  # cv.COLOR_BGR2HSV
RGB2BGR = BGR2RGB # cv.COLOR_RGB2BGR
RGB2HSV = 41  # cv.COLOR_RGB2HSV
BGR2LAB = 44  # cv.COLOR_BGR2LAB
RGB2LAB = 45  # cv.COLOR_RGB2LAB


__all__ = [d for d in dir() if not d.startswith('_')]