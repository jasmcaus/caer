# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

from ._patches import extract_patches_2d

__all__ = (
    'PatchPreprocess'
)

class PatchPreprocess:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def patch_preprocess(self, image):
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]