# Caer -  Modern Computer Vision on the Fly

[![Downloads](https://pepy.tech/badge/caer)](https://pepy.tech/project/caer)
[![license](https://img.shields.io/github/license/jasmcaus/caer?label=license)][license]


## Overview
A lightweight Computer library for high-performance AI research. Caer contains powerful image and video processing operations. 

Caer simplifies your approach towards Computer Vision. It abstracts away unnecessary boilerplate code enabling maximum flexibility. By offering powerful image and video processing algorithms, Caer provides both casual and advanced users with an elegant interface for Machine vision operations.

It leverages the power of libraries like OpenCV and Pillow to speed up your Computer Vision workflow — making it ideal if you want to quickly test out something.

This design philosophy makes Caer ideal for students, researchers, hobbyists and even experts in the fields of Deep Learning and Computer Vision to quickly prototype deep learning models or research ideas.


## Install
See the Caer **[Installation][install]** guide for detailed installation instructions including building from source.

Currently, `caer` supports releases of Python 3.6 onwards; Python 2 is not supported (nor recommended). 
To install the current release:

```shell
$ pip install caer
```


## Getting Started

### Example
```python
>> import caer

# Load a standard 640x427 test image that ships out-of-the-box with caer
>> sunrise = caer.data.sunrise(rgb=True)

# Resize the image to 500x500 while MAINTAINING aspect ratio
>> resized = caer.resize(sunrise, target_size=(500,500), keep_aspect_ratio=True)
```

For more examples, see the [Caer demos](demos).

### Resources

- [**PyPi**](https://pypi.org/project/caer)
- [**Documentation**](https://github.com/jasmcaus/caer/blob/master/docs/README.md)
- [**Issue tracking**](https://github.com/jasmcaus/caer/issues)

## Contributing

We appreciate all contributions. If you plan to contribute new features, utility functions, or extensions to the core, please go through our [Contribution Guidelines][contributing]. By participating, you are expected to uphold the [Code of Conduct][coc].

To contribute, start working through the `caer` codebase, read the [Documentation][docs], navigate to the [Issues][issues] tab and start looking through interesting issues. 

Current contributors can be viewed either from the [Contributors][contributors] file or by using the `caer.__contributors__` command.


All Caer wheels on PyPi are MIT-licensed


[install]: https://github.com/jasmcaus/caer/blob/master/INSTALL.md
[license]: https://github.com/jasmcaus/caer/blob/master/LICENSE