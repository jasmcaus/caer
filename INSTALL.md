## Installing caer

Caer supports an installation of Python 3.6 above, available on Windows, MacOS and Linux systems. 

Version check
------------------------------------------------------------------------------

To see whether `caer` is already installed or to check if an install has
worked, run the following in a Python shell or Jupyter notebook:

```python
>> import caer
>> print(caer.__version__)
```

or, from the command line:

```shell
python -c "import caer; print(caer.__version__)"
```

(Try ``python3`` if ``python`` is unsuccessful.)

You'll see the version number if `caer` is installed and
an error message otherwise.

## Installation via pip and conda
------------------------------------------------------------------------------

These install only `caer` and its dependencies; pip has an option to
include related packages.

### pip

Prerequisites to a pip install: You're able to use your system's command line to
install packages and are using a
[virtual environment](https://towardsdatascience.com/virtual-environments-104c62d48c54?gi=2532aa12906#ee81) (any of[several](https://stackoverflow.com/questions/41573587/what-is-the-difference-between-venv-pyvenv-pyenv-virtualenv-virtualenvwrappe) ).

While it is possible to use pip without a virtual environment, it is not advised: 
virtual environments create a clean Python environment that does not interfere 
with any existing system installation, can be easily removed, and contain only
the package versions your application needs. They help avoid a common
challenge known as 
`dependency hell <https://en.wikipedia.org/wiki/Dependency_hell>`_.

To install the current `caer` you'll need at least Python 3.6. If
your Python is older, pip will find the most recent compatible version.

```shell
# Update pip
>> pip install --upgrade pip
# Install caer (Upgrading if you already have an older version)
>> pip install --upgrade caer
```

To include a selection of other Python packages that expand
``caer``'s capabilities to include, e.g., for Deep Learning, you can install `canaro` as well: 

```shell
>> pip install -upgrade caer[canaro]
```

#### Warning

    Do not use the commands `sudo` and `pip` together as `pip` may
    overwrite critical system libraries which may require you to reinstall your
    operating system.


### conda

Miniconda is a bare-essentials version of the Anaconda package; you'll need to
install packages like `caer` yourself. Like Anaconda, it installs
Python and provides virtual environments.

- [conda documentation](https://docs.conda.io)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Once you have your conda environment set up, you can install `caer`
with the command:

```shell
>> conda install caer
```

## System package managers

Using a package manager (`yum`, `apt-get`, etc.) to install `caer`
or other Python packages is not your best option:

- You're likely to get an older version.

- You'll probably want to make updates and add new packages outside of
  the package manager, leaving you with the same kind of
  dependency conflicts you see when using pip without a virtual environment.

- There's an added risk because operating systems use Python, so if you
  make system-wide Python changes (installing as root or using sudo),
  you can break the operating system.