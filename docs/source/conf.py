# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

#pylint:disable=unused-import,redefined-builtin

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

import sphinx_rtd_theme


# -- Project information -----------------------------------------------------

project = 'Caer'
copyright = '2020, Jason Dsouza'
author = 'Jason Dsouza'

def get_version():
    # find the version number from caer/_meta__.py
    setup_lines = open('../../caer/_meta.py').readlines()
    for l in setup_lines:
        if l.startswith('__version__'):
            version = l.split("=")[-1]
            break
    return version

# The full version, including alpha/beta/rc tags
release = get_version()
master_doc = 'index'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
   	'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    "sphinx.ext.autodoc",
    'sphinx_gallery.gen_gallery',
]

# sphinx_gallery_conf = {
#      'examples_dirs': ['tutorials_source/package', 'tutorials_source/platform'],
#      'gallery_dirs': ['tutorials/package', 'tutorials/platform'],  # path to where to save gallery generated output
#      'filename_pattern': '/tutorial_',
# }

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_type_aliases = None

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
# source_suffix = ['.rst', '.md', '.ipynb']
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
    '.ipynb': 'nbsphinx',
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'collapse_navigation': False, # set to false to prevent menu item collapse
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', '_templates']

# html_favicon = 'favicon.png'

html_css_files = ['main.css']