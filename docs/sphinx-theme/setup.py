import os

from setuptools import setup
from io import open

__version__ = '1.0.0'

def package_files(directory : str):
    """
    Traverses target directory recursivery adding file paths to a list.
    Original solution found at:

        * https://stackoverflow.com/questions/27664504/\
            how-to-add-package-data-recursively-in-python-setup-py

    Parameters
    ----------
    directory: str
        Target directory to traverse.

    Returns
    -------
    paths: list
        List of file paths.
    
    """ 
    paths = []
    for (path, _, files) in os.walk(directory):
        for file in files:
            paths.append(os.path.join('..', path, file))

    return paths


setup(
    name = 'caer_sphinx_theme',
    version =__version__,
    author = 'Jason Dsouza',
    author_email= 'jasmcaus@gmail.com',
    url="https://github.com/jasmcaus/caer-sphinx-theme",
    docs_url="https://github.com/jasmcaus/caer-sphinx-theme",
    description='Caer Sphinx Theme',
    py_modules = ['caer_sphinx_theme'],
    packages = ['caer_sphinx_theme'],
    include_package_data=True,
    zip_safe=False,
    package_data={'caer_sphinx_theme': [
        'theme.conf',
        '*.html',
        'theme_variables.jinja',
        *package_files('caer_sphinx_theme/static')
    ]},
    entry_points = {
        'sphinx.html_themes': [
            'caer_sphinx_theme = caer_sphinx_theme',
        ]
    },
    license= 'MIT License',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Internet",
        "Topic :: Software Development :: Documentation"
    ],
    install_requires=[
       'sphinx'
    ]
)
