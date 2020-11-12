#
#  _____ _____ _____ _____
# |     |     | ___  | __|  Caer - Modern Computer Vision
# |     | ___ |      | \    Languages: Python, C, C++
# |_____|     | ____ |  \   http://github.com/jasmcaus/caer

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


from ._patches import extract_patches_2d


__all__ = [
    'PatchPreprocess'
]


class PatchPreprocess:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def patch_preprocess(self, image):
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]