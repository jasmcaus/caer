from setuptools import setup, find_packages
from configparser import ConfigParser
import io 

VERSION = '1.7.5'

# All settings are in configs.ini
config = ConfigParser(delimiters=['='])
config.read('configs.ini')
cfg = config['DEFAULT']

cfg_keys = 'description keywords author author_email contributors'.split()
expected = cfg_keys + "library_name user git_branch license status min_python audience language".split()
for i in expected: assert i in cfg, f'Missing expected setting: {i}'
setup_cfg = {i:cfg[i] for i in cfg_keys}

NAME = cfg['library_name']
AUTHOR = cfg['author']
AUTHOR_EMAIL = cfg['author_email']
AUTHOR_LONG = AUTHOR + ' <' + AUTHOR_EMAIL + '>'
LICENSE = cfg['license']
URL = cfg['git_url']
DOWNLOAD_URL = cfg['download_url']
PACKAGES = find_packages()
DESCRIPTION = cfg['description']
LONG_DESCRIPTION = io.open('LONG_DESCRIPTION.md', encoding='utf-8').read()
KEYWORDS = cfg['keywords']
REQUIREMENTS = cfg['pip_requirements']
PYTHON_REQUIRES = '>=' + cfg['min_python']
EXTRAS={
        # 'deep': [
        #     'canaro>=1.0.0'
        # ]
        'canaro': 'canaro>=1.0.3'
}
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'License :: OSI Approved :: MIT License',
]
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
        url = URL,
        download_url = DOWNLOAD_URL,
        project_urls={
            'Bug Tracker': URL + '/issues',
            'Documentation': URL + '/blob/master/DOCS.md',
            'Source Code': URL,
        },
        packages=PACKAGES,
        license=LICENSE,
        install_requires=REQUIREMENTS,
        extras_require=EXTRAS,
        python_requires=PYTHON_REQUIRES,
        keywords=KEYWORDS,
        classifiers= [x for x in CLASSIFIERS if x]
    )


if __name__ == '__main__':
    setup_package()