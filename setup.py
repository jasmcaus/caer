#pylint:disable=implicit-str-concat

from setuptools import setup, find_packages
import io 

# Repository on PyPi.org = https://pypi.org/project/caer/

VERSION = '1.7.4'

NAME = 'caer'
AUTHOR = 'Jason Dsouza'
AUTHOR_EMAIL = 'jasmcaus@gmail.com'
AUTHOR_LONG = AUTHOR + ' <' + AUTHOR_EMAIL + '>'
LICENSE = 'MIT'
URL = 'https://github.com/jasmcaus/caer'
DOWNLOAD_URL = 'https://pypi.org/project/caer/'
PACKAGES = find_packages()
KEYWORDS = [
    'computer vision', 'deep learning', 'toolkit', 'image processing', 'video processing','opencv', 'matplotlib'
]
INSTALL_REQUIRES = [
    'numpy', 'opencv-contrib-python', 'h5py'
]
DESCRIPTION = """ A Computer Vision library in Python, built for Humans."""
LONG_DESCRIPTION = io.open('LONG_DESCRIPTION.md', encoding='utf-8').read()
CONTRIBUTORS = io.open('CONTRIBUTORS.md', encoding='utf-8').read()
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'License :: OSI Approved :: MIT License',
]
EXTRAS_REQUIRE={
        'canaro': [
            'tensorflow'
        ]
    },

VERSION_PY_TEXT =\
"""
# This file is automatically generated during the generation of setup.py
# Copyright 2020, Caer
#pylint:disable=syntax-error
author = '%(author)s'
version = '%(version)s'
full_version = '%(full_version)s'
release = %(isrelease)s
contributors = 
%(contributors)s
"""

def write_version(filename='caer/_meta.py'):
    print('[INFO] Writing version.py')
    TEXT = VERSION_PY_TEXT
    FULL_VERSION = VERSION
    ISRELEASED = True

    a = open(filename, 'w')
    try:
        a.write(TEXT % {'author': AUTHOR,
                        'version': VERSION,
                       'full_version': FULL_VERSION,
                       'isrelease': str(ISRELEASED),
                       'contributors': CONTRIBUTORS})
    finally:
        a.close()


def setup_package():
    # Rewrite the version file everytime
    write_version()

    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=AUTHOR,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        url = URL,
        download_url = DOWNLOAD_URL,
        project_urls={
            'Bug Tracker': URL + '/issues',
            'Documentation': URL + '/blob/master/DOCS.md',
            'Source Code': URL,
        },
        packages=PACKAGES,
        license=LICENSE,
        install_requires=INSTALL_REQUIRES,
        extras_requires=EXTRAS_REQUIRE,
        keywords=KEYWORDS,
        classifiers= [x for x in CLASSIFIERS if x]
    )


if __name__ == '__main__':
    setup_package()