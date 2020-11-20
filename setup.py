"""
Caer - Modern Computer Vision on the Fly
=====

A Computer Vision library in Python with powerful image and video processing operations.
Caer is a set of utility functions designed to help speed up your Computer Vision workflow. Functions inside `caer` will help reduce the number of calculation calls your code makes, ultimately making it neat, concise and readable.

Available subpackages
-----------
data
    Test images and example data.
distance
    Distance-computing algorithms
exposure
    Image intensity adjustment, e.g., histogram equalization, etc.
features
    Feature detection and extraction, e.g., texture analysis corners, etc.
filters
    Sharpening, edge finding, rank filters, thresholding, etc.
io
    Reading, saving, and displaying images and video.
morph
    Morphological operations, e.g., opening or skeletonization.
path
    OS-specific Path Manipulations
preprocessing
    Image preprocessing utilities
restoration
    Restoration algorithms, e.g., deconvolution algorithms, denoising, etc.
segmentation
    Partitioning an image into multiple regions.
transform
    Geometric and other transforms, e.g., rotation or the Radon transform.    
util
    Generic utilities.
video
    Video processing utilities


Utilities
---------
__version__
    Caer version string

"""

import sys 
import platform

MAJOR = 1 
MINOR = 9
MICRO = 1
ISRELEASED = True
VERSION = f'{MAJOR}.{MINOR}.{MICRO}'

RUN_CYTHON_BUILD = False

min_version = (3, 6, 1)

def is_right_py_version(min_py_version):
    if sys.version_info < (3,):
        sys.stderr.write('Python 2 has reached end-of-life and is no longer supported by Caer.')
        return False

    if sys.version_info < min_py_version:
        python_min_version_str = '.'.join((str(num) for num in min_py_version))
        no_go = f'You are using Python {platform.python_version()}. Python >={python_min_version_str} is  required.'
        sys.stderr.write(no_go)
        return False

    return True

if not is_right_py_version(min_version):
    sys.exit(-1)


from setuptools import setup, Extension
# from setuptools import find_packages
from distutils.command.build_ext import build_ext
from configparser import ConfigParser

# Configurations

# All settings are in setup.cfg
config = ConfigParser(delimiters=['='])
config.read('setup.cfg')
cfg = config['metadata']
opt = config['options']

cfg_keys = 'description keywords author author_email contributors'.split()
expected = cfg_keys + "name user git_branch license status audience language dev_language".split()
for i in expected: assert i in cfg, f'Missing expected setting: {i}'

def set_builtin(name, value):
    if isinstance(__builtins__, dict):
        __builtins__[name] = value
    else:
        setattr(__builtins__, name, value)

# Prevent numpy from thinking it is still in its setup process:
set_builtin('__NUMPY_SETUP__', False)
import numpy as np 


# Defining Setup Variables

NAME = cfg['name']
AUTHOR = cfg['author']
AUTHOR_EMAIL = cfg['author_email']
AUTHOR_LONG = AUTHOR + ' <' + AUTHOR_EMAIL + '>'
LICENSE = cfg['license']
PLATFORMS = ['Any']
URL = cfg['git_url']
DOWNLOAD_URL = cfg['download_url']
# PACKAGES = find_packages()
PACKAGES = ['caer', 'caer.data', 'caer.path', 'caer.preprocessing', 'caer.utils', 'caer.video']
DESCRIPTION = cfg['description']
LONG_DESCRIPTION = open('LONG_DESCRIPTION.md', encoding='utf-8').read()
KEYWORDS = [i for i in cfg['keywords'].split(', ')]
REQUIREMENTS = [i for i in opt['pip_requirements'].split(', ')]
CLASSIFIERS = [i for i in cfg['classifiers'].split('\n')][1:]
PYTHON_REQUIRES = '>=' + opt['min_python']
EXTENSIONS = {
    'caer.filters.cconvex': ['caer/filters/cconvex.cpp'],
    'caer.filters.cconvolve': ['caer/filters/cconvolve.cpp', 'caer/src/cfilters.cpp'],
    'caer.distance.cdistance': ['caer/distance/cdistance.cpp'],
    'caer.morph.cmorph': ['caer/morph/cmorph.cpp', 'caer/src/cfilters.cpp'],
    'caer.ndi.cndi' : ['caer/ndi/include/cndimage.c', 
                       'caer/ndi/include/cndfilters.c',
                       'caer/ndi/include/cndfourier.c',
                       'caer/ndi/include/cndinterpolation.c',
                       'caer/ndi/include/cndmeasure.c',
                       'caer/ndi/include/cndmorphology.c',
                       'caer/ndi/include/cndsplines.c',
                       'caer/ndi/include/cndsupport.c'
                ]
}
EXT_MODULES = [Extension(key, sources=sources, include_dirs=[np.get_include()]) for key, sources in EXTENSIONS.items()]

STATUSES = [ 
    '1 - Planning', 
    '2 - Pre-Alpha', 
    '3 - Alpha',
    '4 - Beta', 
    '5 - Production/Stable', 
    '6 - Mature', 
    '7 - Inactive' 
]


META_PY_TEXT =\
"""
# This file is automatically generated during the generation of setup.py
# Copyright 2020, Caer
author = '%(author)s'
version = '%(version)s'
full_version = '%(full_version)s'
release = %(isrelease)s
contributors = %(contributors)s
"""


def get_contributors_list(filename='CONTRIBUTORS'):
    contr = [] 
    with open(filename, 'r') as a:
        for line in a:
            line = line.strip()
            # line = """ + line + """
            contr.append(line)
    return contr


def write_meta(filename='caer/_meta.py'):
    print('[INFO] Writing _meta.py')
    TEXT = META_PY_TEXT
    FULL_VERSION = VERSION
    CONTRIBUTORS = get_contributors_list()
    
    a = open(filename, 'w')

    try:
        a.write(TEXT % {'author': AUTHOR_LONG,
                        'version': VERSION,
                       'full_version': FULL_VERSION,
                       'isrelease': str(ISRELEASED),
                       'contributors': CONTRIBUTORS })
    finally:
        a.close()


def get_docs_url():
    return URL + '/blob/master/docs/README.md'


CYTHON_SOURCES = ('',)
def generate_cython():
    import os 
    import subprocess
    cwd = os.path.abspath(os.path.dirname(__file__))
    print('[INFO] Cythonizing sources')
    p = subprocess.call([sys.executable,
                        os.path.join(cwd, 'tools', 'cythonize.py')],
                        cwd=cwd)
    if p != 0:
        raise RuntimeError('[ERROR] Running cythonize failed!')


copt={
    'msvc': ['/EHsc'], 
    'intelw': ['/EHsc']  
}

class build_extension_class(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in copt:
            for e in self.extensions:
                e.extra_compile_args = copt[c]
        build_ext.build_extensions(self)

CMDCLASS = {
    'build_ext': build_extension_class
}


def setup_package():
    # Rewrite the meta file everytime
    write_meta()

    metadata = dict(
        name = NAME,
        version = VERSION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        maintainer = AUTHOR,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        long_description_content_type = 'text/markdown',
        url = URL,
        download_url = DOWNLOAD_URL,
        project_urls = {
            'Bug Tracker': URL + '/issues',
            'Documentation': get_docs_url(),
            'Source Code': URL,
        },
        packages = PACKAGES,
        ext_modules = EXT_MODULES,
        license = LICENSE,
        platforms = PLATFORMS,
        install_requires = REQUIREMENTS,
        python_requires = PYTHON_REQUIRES,
        cmdclass = CMDCLASS,
# Include_package_data is required for setup.py to recognize the MAINFEST.in file
# https://python-packaging.readthedocs.io/en/latest/non-code-files.html
        include_package_data = True,
        zip_safe = False,
        keywords = KEYWORDS,
        classifiers = CLASSIFIERS,
    )

    run_build = RUN_CYTHON_BUILD
    if run_build:
        generate_cython()

    setup(**metadata)


if __name__ == '__main__':
    # Running the build is as simple as: 
    # >> python setup.py sdist bdist_wheel
    # This command includes building the required Python extensions (Cython included)

    # It's recommended, however, to use:
    # >> python setup.py build_ext 
    # first, and then
    # >> python setup.py sdist bdist_wheel
    setup_package()