# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

# Importing the necessary packages
from ._patches import extract_patches_2d

class PatchPreprocess:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def patch_preprocess(self, image):
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]