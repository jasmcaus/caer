# Caer -  Modern Computer Vision on the Fly

A Computer Vision library in Python with powerful image and video processing operations.
Caer is a set of utility functions designed to help speed up your Computer Vision workflow. Functions inside `caer` will help reduce the number of calculation calls your code makes, ultimately making it neat, concise and readable.

[![Downloads](https://pepy.tech/badge/caer)](https://pepy.tech/project/caer)
[![license](https://img.shields.io/github/license/jasmcaus/caer?label=license)][license]


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