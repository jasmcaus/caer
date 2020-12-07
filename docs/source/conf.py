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

#pylint:disable=unused-import,redefined-builtin,redefined-outer-name

import os
import sys
import shutil 

sys.path.insert(0, os.path.abspath('../../'))

import sphinx_rtd_theme

GIT_HERE = os.path.abspath('../../.github')

# -- Project information -----------------------------------------------------

project = 'Caer'
copyright = '2020, Jason Dsouza'
author = 'Jason Dsouza'

def get_version():
    # find the version number from caer/_meta__.py
    f = os.path.abspath('../../caer/_meta.py')
    setup_lines = open(f).readlines()
    version = None 
    for l in setup_lines:
        if l.startswith('version'):
            version = l.split("=")[-1]
            break
    return version

# The full version, including alpha/beta/rc tags
release = get_version()
master_doc = 'index'

# # copy all documents from GH templates like contribution guide
# for f in os.listdir(GIT_HERE):
#     if f.startswith(('CODE_OF_CONDUCT.md','CONTRIBUTING.md')):
#         shutil.copy(os.path.join(GIT_HERE, f), f'community/{f}')


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '2.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.napoleon',
    'sphinx.ext.imgmath',
    'recommonmark',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_paramlinks',
    'sphinx_togglebutton'
]

# napoleon_google_docstring = True
# napoleon_numpy_docstring = False
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = False
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = False
# napoleon_use_rtype = False
# napoleon_type_aliases = None

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# https://berkeley-stat159-f17.github.io/stat159-f17/lectures/14-sphinx..html#conf.py-(cont.)
# https://stackoverflow.com/questions/38526888/embed-ipython-notebook-in-sphinx-document
# I execute the notebooks manually in advance. If notebooks test the code,
# they should be run at build time.
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True
nbsphinx_requirejs_path = ''


# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
# source_suffix = ['.rst', '.md', '.ipynb']
source_suffix = {
    '.rst': 'restructuredtext',
    # '.txt': 'markdown',
    '.md': 'markdown',
    # '.ipynb': 'nbsphinx',
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    'PULL_REQUEST_TEMPLATE.md',
]

# Sphinx will add “permalinks” for each heading and description environment as paragraph signs that
#  become visible when the mouse hovers over them.
# This value determines the text for the permalink; it defaults to "¶". Set it to None or the empty
#  string to disable permalinks.
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_add_permalinks
html_add_permalinks = "¶"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'caer_project': 'docs',
    # 'canonical_url': 'https://jasmcaus.github.io/caer',
    'collapse_navigation': False, # set to false to prevent menu item collapse
    'display_version': True,
    # 'logo_only': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static', '_templates']

# html_favicon = 'favicon.png'

html_css_files = ['main.css']