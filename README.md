# Caer: Computer Vision in Python, built for Humans
A Computer Vision library in Python with powerful image and video processing operations.
Caer is a set of utility functions based off OpenCV, designed to help speed up your Computer Vision workflow. Functions inside `caer` will help reduce the number of calculation calls your code makes, ultimately making it neat, concise and readable.

[![Python](https://img.shields.io/pypi/pyversions/caer.svg?style=plastic)](https://pypi.org/project/caer/)
[![PyPI](https://badge.fury.io/py/caer.svg)](https://pypi.org/project/caer/)

[![Downloads](https://pepy.tech/badge/caer)](https://pepy.tech/project/caer)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/jasmcaus/caer/blob/master/LICENSE)

## Installation
Currently, `caer` supports Python 3; Python is not supported (nor recommended). 
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
pip install -e .
```

You can run the following to verify things installed correctly:

```python
import caer

print(f'Caer version {caer.__version__}')
```

## Resources

- **Homepage:** <https://github.com/jasmcaus/caer>
- **PyPi:** <https://pypi.org/project/caer>
- **Documentation:** <https://github.com/jasmcaus/caer/blob/master/DOCS.md>
- **Issue tracking:** <https://github.com/jasmcaus/caer/issues>

## Contribution Guidelines

If you have improvements to `caer`, send us your pull requests! We'd love to accept your patches! We strongly encourage you to go through the [Contribution Guidelines](https://github.com/jasmcaus/caer/blob/master/.github/CONTRIBUTING.md).

If you want to contribute, start working through the `caer` codebase, read the [Documentation](https://github.com/jasmcaus/caer/blob/master/DOCS.md), navigate to the
[Issues](https://github.com/jasmcaus/caer/issues) tab and start looking through interesting issues. 

Current contributors can be viewed either from the [Contributors](https://github.com/jasmcaus/caer/blob/master/CONTRIBUTORS) file or by using the `caer.__contributors__` command.


## License

Caer is released under the [MIT License](https://github.com/jasmcaus/caer/blob/master/LICENSE).
