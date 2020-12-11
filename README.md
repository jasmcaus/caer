<div align="center">
<!-- ![Caer Logo](https://github.com/jasmcaus/caer/blob/dev/docs/sphinx-theme/_static/thumbs/caer-logo-flat.png) -->
<img src="https://github.com/jasmcaus/caer/blob/dev/docs/sphinx-theme/_static/thumbs/caer-logo-flat.png" alt="Caer Logo" / >

---

[![Python](https://img.shields.io/pypi/pyversions/caer.svg)][py-versions]
[![PyPI](https://badge.fury.io/py/caer.svg)][pypi-latest-version]
[![Downloads](https://pepy.tech/badge/caer)][downloads]
[![Documentation Status](https://readthedocs.org/projects/caer/badge/?version=latest)](https://caer.readthedocs.io/en/latest/?badge=latest)
[![license](https://img.shields.io/github/license/jasmcaus/caer?label=license)][license]

</div>

# Caer - Modern Computer Vision on the Fly

Caer is a lightweight, scalable Computer Vision library for high-performance AI research. It simplifies your approach towards Computer Vision by abstracting away unnecessary boilerplate code giving you maximum flexibility to quickly prototype deep learning models or research ideas. Our design philosophy makes Caer ideal for students, researchers, hobbyists and even experts in the fields of Deep Learning and Computer Vision.


## Overview

Caer is a Python library that consists of the following components:

| Component | Description |
| ---- | --- |
| [**caer**](https://github.com/jasmcaus/caer/) | A lightweight GPU-accelerated Computer Vision library for high-performance AI research |
| [**caer.augment**](https://github.com/jasmcaus/caer/tree/master/caer/augment) | Powerful augmentation functions |
| [**caer.color**](https://github.com/jasmcaus/caer/tree/master/caer/color) | Colorspace operations |
| [**caer.data**](https://github.com/jasmcaus/caer/tree/master/caer/data) | Standard high-quality test images and example data |
| [**caer.path**](https://github.com/jasmcaus/caer/tree/master/caer/path) | OS-specific path manipulations |
| [**caer.preprocessing**](https://github.com/jasmcaus/caer/tree/master/caer/preprocessing) | Image preprocessing utilities. |
| [**caer.transforms**](https://github.com/jasmcaus/caer/tree/master/caer/transforms) | Image transformations |
| [**caer.video**](https://github.com/jasmcaus/caer/tree/master/caer/video) | Video processing utilities |

<!-- | [**caer.utils**](https://github.com/jasmcaus/caer/tree/master/caer/utils) | Generic utilities  | -->
<!-- | [**caer.filters**](https://github.com/jasmcaus/caer/tree/master/caer/filters) | Sharpening, edge finding, rank filters, thresholding, etc | -->

Usually, Caer is used either as:

- a replacement for OpenCV to use the power of GPUs.
- a Computer Vision research platform that provides maximum flexibility and speed.


# Installation 
See the Caer **[Installation][install]** guide for detailed installation instructions including building from source.

Currently, `caer` supports releases of Python 3.6 onwards; Python 2 is not supported (nor recommended). 
To install the current release:

```bash
$ pip install --upgrade caer
```


# Getting Started

## Minimal Example
```python
>> import caer

# Load a standard 640x427 test image that ships out-of-the-box with caer
>> sunrise = caer.data.sunrise(rgb=True)

# Resize the image to 400x400 while MAINTAINING aspect ratio
>> resized = caer.resize(sunrise, target_size=(400,400), preserve_aspect_ratio=True)
```
<img src="examples/thumbs/resize-with-ratio.png" alt="caer.resize()" />

For more examples, see the [Caer demos](https://github.com/jasmcaus/caer/blob/master/examples/).


## Resources

- [**PyPi**](https://pypi.org/project/caer)
- [**Documentation**](https://github.com/jasmcaus/caer/blob/master/docs/README.md)
- [**Issue tracking**](https://github.com/jasmcaus/caer/issues)


# Contributing

We appreciate all contributions, feedback and issues. If you plan to contribute new features, utility functions, or extensions to the core, please go through our [Contribution Guidelines][contributing].

To contribute, start working through the `caer` codebase, read the [Documentation][docs], navigate to the [Issues][issues] tab and start looking through interesting issues. 

Current contributors can be viewed either from the [Contributors][contributors] file or by using the `caer.__contributors__` command.


# License

Caer is open-source and released under the [MIT License](LICENSE).

[contributing]: https://github.com/jasmcaus/caer/blob/master/.github/CONTRIBUTING.md
[docs]: https://caer.rtfd.io
[contributors]: https://github.com/jasmcaus/caer/blob/master/CONTRIBUTORS
[coc]: https://github.com/jasmcaus/caer/blob/master/CODE_OF_CONDUCT.md
[issues]: https://github.com/jasmcaus/caer/issues
[install]: https://github.com/jasmcaus/caer/blob/master/INSTALL.md
[demos]: https://github.com/jasmcaus/caer/blob/master/examples/

[downloads]: https://pepy.tech/project/caer
[py-versions]: https://pypi.org/project/caer/
[pypi-latest-version]: https://pypi.org/project/caer/
[license]: https://github.com/jasmcaus/caer/blob/master/LICENSE