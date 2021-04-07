<div align="center">
<!-- ![Caer Logo](https://github.com/jasmcaus/caer/blob/dev/docs/sphinx-theme/_static/thumbs/caer-logo-flat.png) -->
<img src="https://github.com/jasmcaus/caer/blob/dev/docs/sphinx-theme/_static/thumbs/caer-logo-flat.png" alt="Caer Logo" / >

---

[![Python](https://img.shields.io/pypi/pyversions/caer.svg)][py-versions]
[![PyPI](https://badge.fury.io/py/caer.svg)][pypi-latest-version]
[![Twitter](https://img.shields.io/twitter/follow/jasmcaus.svg?style=flatl&label=Follow&logo=twitter&logoColor=white&color=1da1f2)][twitter-badge]
[![Downloads](https://pepy.tech/badge/caer)][downloads]
[![ReadTheDocs](https://readthedocs.org/projects/caer/badge/?version=latest)][docs]
[![license](https://img.shields.io/github/license/jasmcaus/caer?label=license)][license]

</div>

# Caer - Modern Computer Vision on the Fly

Caer is a *lightweight, high-performance* Vision library for high-performance AI research. We wrote this framework to simplify your approach towards Computer Vision by abstracting away unnecessary boilerplate code giving you the **flexibility** to quickly prototype deep learning models and research ideas. The end result is a library quite different in its design, that’s easy to understand, plays well with others, and is a lot of fun to use.

Our elegant, *type-checked* API and design philosophy makes Caer ideal for students, researchers, hobbyists and even experts in the fields of Deep Learning and Computer Vision.


## Overview

Caer is a Python library that consists of the following components:

| Component | Description |
| ---- | --- |
| [**caer**](https://github.com/jasmcaus/caer/) | A lightweight GPU-accelerated Computer Vision library for high-performance AI research |
| [**caer.color**](https://github.com/jasmcaus/caer/tree/master/caer/color) | Colorspace operations |
| [**caer.data**](https://github.com/jasmcaus/caer/tree/master/caer/data) | Standard high-quality test images and example data |
| [**caer.path**](https://github.com/jasmcaus/caer/tree/master/caer/path) | OS-specific path manipulations |
| [**caer.preprocessing**](https://github.com/jasmcaus/caer/tree/master/caer/preprocessing) | Image preprocessing utilities. |
| [**caer.transforms**](https://github.com/jasmcaus/caer/tree/master/caer/transforms) | Powerful image transformations and augmentations |
| [**caer.video**](https://github.com/jasmcaus/caer/tree/master/caer/video) | Video processing utilities |

<!-- | [**caer.utils**](https://github.com/jasmcaus/caer/tree/master/caer/utils) | Generic utilities  | -->
<!-- | [**caer.filters**](https://github.com/jasmcaus/caer/tree/master/caer/filters) | Sharpening, edge finding, rank filters, thresholding, etc | -->

Usually, Caer is used either as:

- a replacement for OpenCV to use the power of GPUs.
- a Computer Vision research platform that provides maximum flexibility and speed.


# Installation 
See the Caer **[Installation][install]** guide for detailed installation instructions (including building from source).

Currently, `caer` supports releases of Python 3.6 onwards; Python 2 is not supported (nor recommended). 
To install the current release:

```shell
$ pip install --upgrade caer
```


# Getting Started

## Minimal Example
```python
import caer

# Load a standard 640x427 test image that ships out-of-the-box with caer
sunrise = caer.data.sunrise(rgb=True)

# Resize the image to 400x400 while MAINTAINING aspect ratio
resized = caer.resize(sunrise, target_size=(400,400), preserve_aspect_ratio=True)
```
<img src="examples/thumbs/resize-with-ratio.png" alt="caer.resize()" />

For more examples, see the [Caer demos](https://github.com/jasmcaus/caer/blob/master/examples/) or [Read the documentation](http://caer.rtfd.io)


# Resources

- [**PyPi**](https://pypi.org/project/caer)
- [**Documentation**](https://github.com/jasmcaus/caer/blob/master/docs/README.md)
- [**Issue tracking**](https://github.com/jasmcaus/caer/issues)


# Contributing

We appreciate all contributions, feedback and issues. If you plan to contribute new features, utility functions, or extensions to the core, please go through our [Contribution Guidelines][contributing].

To contribute, start working through the `caer` codebase, read the [Documentation][docs], navigate to the [Issues][issues] tab and start looking through interesting issues. 

Current contributors can be viewed either from the [Contributors][contributors] file or by using the `caer.__contributors__` command.


# Asking for help
If you have any questions, please:
1. [Read the docs](https://caer.rtfd.io/en/latest/).
2. [Look it up in our Github Discussions (or add a new question)](https://github.com/jasmcaus/caer/discussions).
2. [Search through the issues](https://github.com/jasmcaus/caer/issues).


# License

Caer is open-source and released under the [MIT License](LICENSE).


# BibTeX
If you want to cite the framework feel free to use this (but only if you loved it 😊):

```bibtex
@article{jasmcaus,
  title={Caer},
  author={Dsouza, Jason},
  journal={GitHub. Note: https://github.com/jasmcaus/caer},
  volume={2},
  year={2020-2021}
}
```

[contributing]: https://github.com/jasmcaus/caer/blob/master/.github/CONTRIBUTING.md
[docs]: https://caer.rtfd.io
[contributors]: https://github.com/jasmcaus/caer/blob/master/CONTRIBUTORS
[coc]: https://github.com/jasmcaus/caer/blob/master/CODE_OF_CONDUCT.md
[issues]: https://github.com/jasmcaus/caer/issues
[install]: https://github.com/jasmcaus/caer/blob/master/INSTALL.md
[demos]: https://github.com/jasmcaus/caer/blob/master/examples/

[twitter-badge]: https://twitter.com/jasmcaus
[downloads]: https://pepy.tech/project/caer
[py-versions]: https://pypi.org/project/caer/
[pypi-latest-version]: https://pypi.org/project/caer/
[license]: https://github.com/jasmcaus/caer/blob/master/LICENSE
