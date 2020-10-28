from setuptools import setup, find_packages
from configparser import ConfigParser
import io 
import sys 
import platform

# v1.7.6 cannot be used (previously deleted)
VERSION = '1.7.8'


# Checking if right Python version is used

min_version = (3, 6, 1)
# max_version = (3, 9, 0)

# def is_right_py_version(min_py_version, max_py_version):
def is_right_py_version(min_py_version):
    python_min_version_str = '.'.join((str(num) for num in min_py_version))
    # python_max_version_str = '.'.join((str(num) for num in max_py_version))
    # if sys.version_info < min_py_version or sys.version_info >= max_py_version:
    if sys.version_info < min_py_version:
        # no_go = 'You are using Python {}. Python >={},<{} is required.'.format(platform.python_version(), python_min_version_str, python_max_version_str)
        no_go = 'You are using Python {}. Python >={} is required.'.format(platform.python_version(), python_min_version_str)
        sys.stderr.write(no_go)
        return False
    return True

# if not is_right_py_version(min_version, max_version):
if not is_right_py_version(min_version):
    sys.exit(-1)


# Configurations

# All settings are in setup.cfg
config = ConfigParser(delimiters=['='])
config.read('setup.cfg')
cfg = config['metadata']
opt = config['options']

cfg_keys = 'description keywords author author_email contributors'.split()
expected = cfg_keys + "name user git_branch license status audience language dev_language".split()
for i in expected: assert i in cfg, f'Missing expected setting: {i}'
# setup_cfg = {i:cfg[i] for i in cfg_keys}


# Defining Setup Variables

NAME = cfg['name']
AUTHOR = cfg['author']
AUTHOR_EMAIL = cfg['author_email']
AUTHOR_LONG = AUTHOR + ' <' + AUTHOR_EMAIL + '>'
LICENSE = cfg['license']
URL = cfg['git_url']
DOWNLOAD_URL = cfg['download_url']
PACKAGES = find_packages()
DESCRIPTION = cfg['description']
LONG_DESCRIPTION = io.open('LONG_DESCRIPTION.md', encoding='utf-8').read()
KEYWORDS = [i for i in cfg['keywords'].split(', ')]
REQUIREMENTS = [i for i in opt['pip_requirements'].split(', ')]
CLASSIFIERS = [i for i in cfg['classifiers'].split('\n')][1:]
PYTHON_REQUIRES = '>=' + opt['min_python']
EXTRAS={
        'canaro': 'canaro>=1.0.6'
}
STATUSES = [ 
    '1 - Planning', 
    '2 - Pre-Alpha', 
    '3 - Alpha',
    '4 - Beta', 
    '5 - Production/Stable', 
    '6 - Mature', 
    '7 - Inactive' 
]

VERSION_PY_TEXT =\
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
    TEXT = VERSION_PY_TEXT
    FULL_VERSION = VERSION
    ISRELEASED = True
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


def setup_package():
    # Rewrite the meta file everytime
    write_meta()

    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=AUTHOR,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type='text/markdown',
        url=URL,
        download_url=DOWNLOAD_URL,
        project_urls={
            'Bug Tracker': URL + '/issues',
            'Documentation': URL + '/blob/master/docs/DOCS.md',
            'Source Code': URL,
        },
        packages=PACKAGES,
        license=LICENSE,
        install_requires=REQUIREMENTS,
        extras_require=EXTRAS,
        python_requires=PYTHON_REQUIRES,
        keywords=KEYWORDS,
        classifiers= CLASSIFIERS,
# Include_package_data is required for setup.py to recognize the MAINFEST.in file
# https://python-packaging.readthedocs.io/en/latest/non-code-files.html
        include_package_data=True
    )


if __name__ == '__main__':
    setup_package()