import sys 
import platform
import warnings
import textwrap


MAJOR = 1 
MINOR = 7
MICRO = 8
ISRELEASED = True
VERSION = f'{MAJOR}.{MINOR}.{MICRO}'

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


from setuptools import setup, find_packages
from configparser import ConfigParser
import io 


# Configurations

# All settings are in setup.cfg
config = ConfigParser(delimiters=['='])
config.read('setup.cfg')
cfg = config['metadata']
opt = config['options']

cfg_keys = 'description keywords author author_email contributors'.split()
expected = cfg_keys + "name user git_branch license status audience language dev_language".split()
for i in expected: assert i in cfg, f'Missing expected setting: {i}'


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
    # if not ISRELEASED:
    #     return "https://caer.com/devdocs"
    # else:
    #     # For releaeses, this URL ends up on pypi.
    #     # By pinning the version, users looking at old PyPI releases can get
    #     # to the associated docs easily.
    #     return "https://caer.com/doc/{}.{}".format(MAJOR, MINOR)
    return URL + '/blob/master/docs/README.md'


# def generate_cython():
#     cwd = os.path.abspath(os.path.dirname(__file__))
#     print("Cythonizing sources")
#     for d in ('random',):
#         p = subprocess.call([sys.executable,
#                              os.path.join(cwd, 'tools', 'cythonize.py'),
#                              'caer/{0}'.format(d)],
#                             cwd=cwd)
#         if p != 0:
#             raise RuntimeError("Running cythonize failed!")


def parse_setuppy_commands():
    """Check the commands and respond appropriately.  Disable broken commands.

    Return a boolean value for whether or not to run the build or not (avoid
    parsing Cython and template files if False).
    """
    args = sys.argv[1:]

    if not args:
        # User forgot to give an argument probably, let setuptools handle that.
        return True

    info_commands = ['--help-commands', '--name', '--version', '-V',
                     '--fullname', '--author', '--author-email',
                     '--maintainer', '--maintainer-email', '--contact',
                     '--contact-email', '--url', '--license', '--description',
                     '--long-description', '--platforms', '--classifiers',
                     '--keywords', '--provides', '--requires', '--obsoletes']

    for command in info_commands:
        if command in args:
            return False

    # Note that 'alias', 'saveopts' and 'setopt' commands also seem to work
    # fine as they are, but are usually used together with one of the commands
    # below and not standalone.  Hence they're not added to good_commands.
    good_commands = ('develop', 'sdist', 'build', 'build_ext', 'build_py',
                     'build_clib', 'build_scripts', 'bdist_wheel', 'bdist_rpm',
                     'bdist_wininst', 'bdist_msi', 'bdist_mpkg', 'build_src')

    for command in good_commands:
        if command in args:
            return True

    # The following commands are supported, but we need to show more
    # useful messages to the user
    if 'install' in args:
        print(textwrap.dedent("""
            Note: if you need reliable uninstall behavior, then install
            with pip instead of using `setup.py install`:

              - `pip install .`       (from a git repo or downloaded source
                                       release)
              - `pip install numpy`   (last NumPy release on PyPi)

            """))
        return True

    if '--help' in args or '-h' in sys.argv[1]:
        print(textwrap.dedent("""
            NumPy-specific help
            -------------------

            To install NumPy from here with reliable uninstall, we recommend
            that you use `pip install .`. To install the latest NumPy release
            from PyPi, use `pip install numpy`.

            For help with build/installation issues, please ask on the
            numpy-discussion mailing list.  If you are sure that you have run
            into a bug, please report it at https://github.com/numpy/numpy/issues.

            Setuptools commands help
            ------------------------
            """))
        return False

    # The following commands aren't supported.  They can only be executed when
    # the user explicitly adds a --force command-line argument.
    bad_commands = dict(
        test="""
            `setup.py test` is not supported.  Use one of the following
            instead:

              - `python runtests.py`              (to build and test)
              - `python runtests.py --no-build`   (to test installed numpy)
              - `>>> numpy.test()`           (run tests for installed numpy
                                              from within an interpreter)
            """,
        upload="""
            `setup.py upload` is not supported, because it's insecure.
            Instead, build what you want to upload and upload those files
            with `twine upload -s <filenames>` instead.
            """,
        upload_docs="`setup.py upload_docs` is not supported",
        easy_install="`setup.py easy_install` is not supported",
        clean="""
            `setup.py clean` is not supported, use one of the following instead:

              - `git clean -xdf` (cleans all files)
              - `git clean -Xdf` (cleans all versioned files, doesn't touch
                                  files that aren't checked into the git repo)
            """,
        check="`setup.py check` is not supported",
        register="`setup.py register` is not supported",
        bdist_dumb="`setup.py bdist_dumb` is not supported",
        bdist="`setup.py bdist` is not supported",
        build_sphinx="""
            `setup.py build_sphinx` is not supported, use the
            Makefile under doc/""",
        flake8="`setup.py flake8` is not supported, use flake8 standalone",
        )
    bad_commands['nosetests'] = bad_commands['test']
    for command in ('upload_docs', 'easy_install', 'bdist', 'bdist_dumb',
                    'register', 'check', 'install_data', 'install_headers',
                    'install_lib', 'install_scripts', ):
        bad_commands[command] = "`setup.py %s` is not supported" % command

    for command in bad_commands.keys():
        if command in args:
            print(textwrap.dedent(bad_commands[command]) +
                  "\nAdd `--force` to your command to use it anyway if you "
                  "must (unsupported).\n")
            sys.exit(1)

    # Commands that do more than print info, but also don't need Cython and
    # template parsing.
    other_commands = ['egg_info', 'install_egg_info', 'rotate']
    for command in other_commands:
        if command in args:
            return False

    # If we got here, we didn't detect what setup.py command was given
    warnings.warn("Unrecognized setuptools command, proceeding with "
                  "generating Cython sources and expanding templates",
                  stacklevel=2)
    return True


def setup_package():
    # Rewrite the meta file everytime
    write_meta()

    setup(
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
        license = LICENSE,
        install_requires = REQUIREMENTS,
        extras_require = EXTRAS,
        python_requires = PYTHON_REQUIRES,
        include_package_data = True,
        zip_safe = False,
        keywords = KEYWORDS,
        classifiers = CLASSIFIERS,
# Include_package_data is required for setup.py to recognize the MAINFEST.in file
# https://python-packaging.readthedocs.io/en/latest/non-code-files.html
    )


if __name__ == '__main__':
    setup_package()