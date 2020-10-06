# Caer: Computer Vision in Python, built for Humans
A Computer Vision library in Python with powerful image and video processing operations.

[![Python](https://img.shields.io/pypi/pyversions/caer.svg?style=plastic)](https://pypi.org/project/caer/)
[![PyPI](https://badge.fury.io/py/caer.svg)](https://pypi.org/project/caer/)

[![Downloads](https://pepy.tech/badge/caer)](https://pepy.tech/project/caer)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/jasmcaus/caer/blob/master/LICENSE)

## Installation
To install the current release:

```shell
$ pip install caer
```

Optionally, Caer can also install [canaro](https://github.com/jasmcaus/canaro) if you install it with `pip install caer[canaro]`

### Installing from Source
First, clone the repo on your machine and then install with `pip`:

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

- **Homepage:** <https://github.com/jasmcaus/caer/>
- **PyPi:** <https://pypi.org/project/caer/>
- **Docs:** <https://github.com/jasmcaus/caer/blob/master/DOCS.md>
- **Issue tracking:** <https://github.com/jasmcaus/caer/issues>

## Contribution Guidelines

If you have improvements to `caer`, send us your pull requests! We'd love to accept your patches! We strongly encourage you to go through the [Contribution Guidelines](CONTRIBUTING.md).

If you want to contribute, start working through the `caer` codebase, navigate to the
[Issues](https://github.com/jasmcaus/caer/issues) tab and start looking through interesting issues. 

Current contributors can be viewed either from the [CONTRIBUTORS](https://github.com/jasmcaus/caer/blob/master/CONTRIBUTORS) file or by using the `caer.__contributors__` command


## License

Caer is released under the [MIT License](https://github.com/jasmcaus/caer/blob/master/LICENSE)