#
#  _____ _____ _____ _____
# |     |     | ___  | __|  Caer - Modern Computer Vision
# |     | ___ |      | \    Languages: Python, C, C++
# |_____|     | ____ |  \   http://github.com/jasmcaus/caer

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>

# We don't want to modify the root dir in any way 
# This gets distrupted if this file is called directly
# Checks are in place:
if __name__ != '__main__':
    from .path import dirname

    __curr__ = dirname(__file__).replace("\\", "/")