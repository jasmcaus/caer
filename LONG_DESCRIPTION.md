A Computer Vision library in Python with powerful image processing operations, including support for Deep Learning models built using the Keras framework

[![Downloads](https://pepy.tech/badge/caer)](https://pepy.tech/project/caer)
[![license](https://img.shields.io/github/license/jasmcaus/caer?label=license)][license]


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
>> import caer
>> print(f'Caer version: {caer.__version__}')
```


## Resources

- [**Homepage**](https://github.com/jasmcaus/caer)
- [**Documentation**](https://github.com/jasmcaus/caer/blob/master/docs/README.md)
- [**Issue tracking**](https://github.com/jasmcaus/caer/issues)


All Caer wheels on PyPi are MIT-licensed


[install]: https://github.com/jasmcaus/caer/blob/master/INSTALL.md
[license]: https://github.com/jasmcaus/caer/blob/master/LICENSE