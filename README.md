# Caer: Computer Vision on the Fly
[![Python](https://img.shields.io/pypi/pyversions/caer.svg)][py-versions]
[![PyPI](https://badge.fury.io/py/caer.svg)][pypi-latest-version]
[![Downloads](https://pepy.tech/badge/caer)][downloads]
[![license](https://img.shields.io/github/license/jasmcaus/caer?label=license)][license]

A Computer Vision library in Python with powerful image and video processing operations.
Caer is a set of utility functions designed to help speed up your Computer Vision workflow. Functions inside `caer` will help reduce the number of calculation calls your code makes, ultimately making it neat, concise and readable.


# About Caer

Caer is a Python library that consists of the following components:

| Component | Description |
| ---- | --- |
| [**caer**](https://github.com/jasmcaus/caer/) | A powerful Computer Vision library |
| [**caer.data**](https://github.com/jasmcaus/caer/tree/master/caer/data) | Standard high-quality test images |
| [**caer.filters**](https://github.com/jasmcaus/caer/tree/master/caer/filters) | Advanced Image Filters |
| [**caer.utils**](https://github.com/jasmcaus/caer/tree/master/caer/utils) | General utilities  |
| [**caer.path**](https://github.com/jasmcaus/caer/tree/master/caer/path) | OS-specific path manipulations |
| [**caer.preprocessing**](https://github.com/jasmcaus/caer/tree/master/caer/preprocessing) | Image preprocessing utilities. |
| [**caer.video**](https://github.com/jasmcaus/caer/tree/master/caer/video) | Video processing utilities |

Usually, Caer is used either as:

- a replacement for OpenCV to use the power of GPUs.
- a Computer Vision research platform that provides maximum flexibility and speed.


## Installation
See **[Installation][install]** for detailed installation instructions. 


Currently, `caer` supports releases of Python 3.6 onwards; Python 2 is not supported (nor recommended). 
To install the current release:

```shell
$ pip install caer
```

Optionally, Caer can also install [canaro](https://github.com/jasmcaus/canaro) if you install it with `pip install caer[canaro]`

### From Source
If you plan to develop `caer` yourself, or want to be on the cutting edge, you can use an editable install:

```shell
git clone https://github.com/jasmcaus/caer.git
cd caer
pip install -e . # Do this once to add the package to the Python Path
```

You can run the following to verify things installed correctly:

```python
>> import caer
>> print(f'Caer version: {caer.__version__}')
```


# Getting Started

## Resources

- [**PyPi**](https://pypi.org/project/caer)
- [**Documentation**](https://github.com/jasmcaus/caer/blob/master/docs/README.md)
- [**Issue tracking**](https://github.com/jasmcaus/caer/issues)

## Contributing

We appreciate all contributions. If you plan to contribute new features, utility functions, or extensions to the core, please go through our [Contribution Guidelines][contributing]. By participating, you are expected to uphold the [Code of Conduct](https://github.com/jasmcaus/caer/blob/master/CODE_OF_CONDUCT.md)

To contribute, start working through the `caer` codebase, read the [Documentation][docs], navigate to the [Issues][issues] tab and start looking through interesting issues. 

Current contributors can be viewed either from the [Contributors][contributors] file or by using the `caer.__contributors__` command.


## License

Caer is released under the [MIT License](https://github.com/jasmcaus/caer/blob/master/LICENSE).

[contributing]: https://github.com/jasmcaus/caer/blob/master/.github/CONTRIBUTING.md
[docs]: https://github.com/jasmcaus/caer/blob/master/docs/README.md
[contributors]: https://github.com/jasmcaus/caer/blob/master/CONTRIBUTORS
[issues]: https://github.com/jasmcaus/caer/issues
[install]: https://github.com/jasmcaus/caer/blob/master/INSTALL.md

[downloads]: https://pepy.tech/project/caer
[py-versions]: https://pypi.org/project/caer/
[pypi-latest-version]: https://pypi.org/project/caer/
[license]: https://github.com/jasmcaus/caer/blob/master/LICENSE