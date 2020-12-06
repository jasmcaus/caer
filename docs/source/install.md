## Installing caer

Caer supports an installation of Python 3.6 above, available on Windows, MacOS and Linux systems. 

Version check
------------------------------------------------------------------------------

To see whether `caer` is already installed or to check if an install has worked, run the following in a Python shell or Jupyter notebook:

```python
>> import caer
>> print(caer.__version__)
```

or, from the command line:

```shell
python -c "import caer; print(caer.__version__)"
```

(Try ``python3`` if ``python`` is unsuccessful.)

You'll see the version number if `caer` is installed and an error message otherwise.

&thinsp;

## Installation

### pip (Recommended)

Prerequisites to a pip install: You are able to use your system's command line to install packages and are using a [virtual environment](https://towardsdatascience.com/virtual-environments-104c62d48c54?gi=2532aa12906#ee81) (any of [several](https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe)).

To install the current `caer` you'll need at least Python 3.6.1. If you have an older version of Python, you will not be able to use `caer`.

```shell
$ pip install --upgrade caer
```

Alternatively, you may download the wheels from [PyPi](https://pypi.org/project/caer/#files)

<!-- To include a selection of other Python packages that expand `caer`'s capabilities, e.g., for Deep Learning, you can install `canaro` as well: 

```shell
$ pip install --upgrade caer[canaro]
``` -->

#### Warning

Do not use the commands `sudo` and `pip` together as `pip` may overwrite critical system libraries which may require you to reinstall your operating system.


### Bleeding Edge 
If a bug fix was made in the repo and you can't wait till a new release is made, you can install the bleeding edge version of `caer` using:
```python
pip install git+https://github.com/jasmcaus/caer.git
```
    

### From Source
If you plan to develop `caer` yourself, or want to be on the cutting edge, you can use an editable install:

First, uninstall any existing installations:
```shell
pip uninstall -y caer
```

Clone the repo:
```shell
git clone https://github.com/jasmcaus/caer.git
cd caer
pip install -e . # Do this once to add the package to the Python Path
```

To update the installation:
```shell
git pull  # Grabs the latest source
pip install -e . # Reinstalls Caer
```

&thinsp;

## System package managers

Using a package manager (`yum`, `apt-get`, etc.) to install `caer` or other Python packages is not your best option:

- You're likely to get an older version.

- You'll probably want to make updates and add new packages outside of the package manager, leaving you with the same kind ofdependency conflicts you see when using pip without a virtual environment.

- There's an added risk because operating systems use Python, so if you make system-wide Python changes (installing as root or using sudo), you can break the operating system.