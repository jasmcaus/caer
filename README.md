# Caer: Computer Vision in Python, built for Humans
A Computer Vision library in Python with powerful image and video processing operations.
Caer is a set of utility functions based off OpenCV, designed to help speed up your Computer Vision workflow. Functions inside `caer` will help reduce the number of calculation calls your code makes, ultimately making it neat, concise and readable.

[![Python](https://img.shields.io/pypi/pyversions/caer.svg?style=plastic)][py-versions]
[![PyPI](https://badge.fury.io/py/caer.svg)][pypi-latest-version]

[![Downloads](https://pepy.tech/badge/caer)][downloads]
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)][license]

## Installation
See [Installation][install] for detailed installation instructions. 


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
import caer

print(f'Caer version: {caer.__version__}')
```

## Resources

- **Homepage:** <https://github.com/jasmcaus/caer>
- **PyPi:** <https://pypi.org/project/caer>
- **Documentation:** <https://github.com/jasmcaus/caer/blob/master/DOCS.md>
- **Issue tracking:** <https://github.com/jasmcaus/caer/issues>

## Contributing

We appreciate all contributions. If you plan to contribute new features, utility functions, or extensions to the core, first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the core in a different direction than you might be aware of.

To learn more about making a contribution to Caer, please go through our [Contribution Guidelines][contributing].

If you want to contribute, start working through the `caer` codebase, read the [Documentation][docs], navigate to the [Issues][issues] tab and start looking through interesting issues. 

Current contributors can be viewed either from the [Contributors][contributors] file or by using the `caer.__contributors__` command.


## License

Caer is released under the [MIT License](https://github.com/jasmcaus/caer/blob/master/LICENSE).

[contributing]: https://github.com/jasmcaus/caer/blob/master/.github/CONTRIBUTING.md
[docs]: https://github.com/jasmcaus/caer/blob/master/DOCS.md
[contributors]: https://github.com/jasmcaus/caer/blob/master/CONTRIBUTORS
[issues]: https://github.com/jasmcaus/caer/issues
[install]: https://github.com/jasmcaus/caer/blob/master/INSTALL.md

[downloads]: https://pepy.tech/project/caer
[py-versions]: https://pypi.org/project/caer/
[pypi-latest-version]: https://pypi.org/project/caer/
[license]: https://github.com/jasmcaus/caer/blob/master/LICENSE