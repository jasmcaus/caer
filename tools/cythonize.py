""" cythonize
    Cythonize pyx files into C files as needed.
    Usage: cythonize [root_dir]
    Default [root_dir] is '../caer'.
    
    Simple script to invoke Cython on all .pyx files; while waiting for a proper build system. 

    For now, this script should be run by developers when changing Cython files
    only, and the resulting C files checked in, so that end-users (and Python-only
    developers) do not get the Cython dependencies.
    
    Inspired by the script originally written by Dag Sverre Seljebotn, and copied here from:
    https://raw.github.com/dagss/private-scipy-refactor/cythonize/cythonize.py
    Note: this script does not check any of the dependent C libraries; it only
    operates on the Cython .pyx files.

    The above script compares pyx files to see if they have been changed relative to their corresponding C files by comparing hashes stored in a database file. It calls `cython [file.pyx]` which merely converts the .pyx files to .c files. This cannot be imported into a Python file (an extension needs to be built for that). 

    This current script uses a command which converts the .pyx --> .c which then builds the required extensions (.pyd on Windows).

    To manually build the required extensions from .c files, use the following code:
    
    # from distutils.core import setup, Extension

    # module1 = Extension('demo',
    #                     include_dirs = ['/usr/local/include'],
    #                     libraries = ['tcl83'],
    #                     library_dirs = ['/usr/local/lib'],
    #                     sources = ['demo.c'])

    # setup (
        # # name = 'PackageName',
        # # version = '1.0',
        # # description = 'This is a demo package',
        # # author = 'Martin v. Loewis',
        # # author_email = 'martin@v.loewis.de',
        # # url = 'https://docs.python.org/extending/building',
        # # long_description = '''
        # #       This is really just a demo package.
        # #       ''',
    #     ext_modules = [module1])
"""


import os 
import sys 
import datetime
import subprocess 

#pylint:disable=redefined-builtin, bare-except

ROOT_DIR = '../caer'

CYTHON_SOURCES = []

# WindowsError is not defined on unix systems
try:
    WindowsError
except NameError:
    WindowsError = None

# This is a bit hackish. We are auto-generating a python file that will allow for Cython to build the required extensions.
SETUP_TEXT =\
"""
# DO NOT EDIT!
# THIS FILE WAS AUTOMATICALLY GENERATED AT %(time)s DURING THE CYTHON BUILD CYCLE OF ALL .PYX FILES
# Copyright 2020-2021, Caer

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(%(source)s),
    zip_safe=False,
)
"""


#
# Rules
#
def find_files(path, *ext):
    for root, _, files in os.walk(path):
        if 'caer' in root and not '.git' in root:
            for file in files:
                if file.endswith(ext):
                    base = os.path.join(root,file).replace('\\', '/')
                    abspath = os.path.abspath(base)
                    CYTHON_SOURCES.append(abspath)


def normpath(path):
    path = path.replace(os.sep, '/')
    if path.startswith('./'):
        path = path[2:]
    elif path.startswith('../'):
        path = path[3:]
    return path


def process_pyx():
    flags = ['-3', '--inplace']
    build_cython = True

    try:
        from Cython.Compiler.Version import version as cython_version
    except ImportError:
        raise OSError('[ERROR] Cython needs to be installed')

    # Cython 0.29.21 is required for Python 3.9 and there are other fixes in the 0.29 series that are needed for earlier Python versions.
    # Note: keep in sync with that in pyproject.toml
    pyproject_toml = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
    if not os.path.exists(pyproject_toml):
        raise RuntimeError('[ERROR] pyproject.toml was not found. Ensure it was not deleted')

    # Try to find the minimum version from pyproject.toml
    # required_cython_version = '0.29.21'
    with open(pyproject_toml) as f:
        for line in f:
            if 'cython' not in line.lower():
                continue
            _, line = line.split('=')
            required_cython_version, _ = line.split("'")
            break
        else:
            raise RuntimeError('[ERROR] An appropriate version for Cython could not be found from pyproject.toml')

    if cython_version < required_cython_version:
        raise RuntimeError(f'[ERROR] Building Caer requires Cython >= {required_cython_version}')

    # Populating CYTHON_SOURCES
    find_files(ROOT_DIR, '.pyx')

    if len(CYTHON_SOURCES) == 0:
        sys.stderr.write('[ERROR] No Cython files found matching the required extensions. Cython build escaping\n')
        build_cython = False

    if build_cython:
        # Writing to build_cython.py (temp)
        a = open('tools/build_cython.py', 'w')
        try:
            a.write(SETUP_TEXT % {'time': datetime.date.today().strftime("%B %d, %Y") ,
                                  'source': CYTHON_SOURCES } )
        finally:
            a.close()


        # Can only concatenate lists
        # build_cython.py is an auto-generated Python script by cythonize.py
        # Do not run cythonize.py directly unless the .c files need to be re-compiled
        try:
            r = subprocess.check_call(['python' + 'build_cython.py' + 'build_ext'] + flags)

            if r != 0:
                p = subprocess.check_call([sys.executable, '-m', 'build_cython.py'] + flags)

                if p != 0:
                    raise Exception()

        except:
            sys.stderr.write('[ERROR] Building cython failed')



def main():
    # try:
    #     ROOT_DIR = sys.argv[1]
    # except IndexError:
    #     ROOT_DIR = ROOT_DIR
    process_pyx()


if __name__ == '__main__':
    main()




# """ cythonize
# Cythonize pyx files into C files as needed.
# Usage: cythonize [root_dir]
# Default [root_dir] is 'caer'.
# Checks pyx files to see if they have been changed relative to their
# corresponding C files.  If they have, then runs cython on these files to
# recreate the C files.
# The script thinks that the pyx files have changed relative to the C files
# by comparing hashes stored in a database file.
# Simple script to invoke Cython (and Tempita) on all .pyx (.pyx.in)
# files; while waiting for a proper build system. Uses file hashes to
# figure out if rebuild is needed.
# For now, this script should be run by developers when changing Cython files
# only, and the resulting C files checked in, so that end-users (and Python-only
# developers) do not get the Cython/Tempita dependencies.
# Originally written by Dag Sverre Seljebotn, and copied here from:
# https://raw.github.com/dagss/private-scipy-refactor/cythonize/cythonize.py
# Note: this script does not check any of the dependent C libraries; it only
# operates on the Cython .pyx files.
# """
# #pylint:disable=redefined-builtin

# import os
# import re
# import sys
# import hashlib
# import subprocess

# HASH_FILE = 'cythonize.dat'
# ROOT_DIR = 'caer'
# VENDOR = 'caer'

# CYTHON_SOURCES = []

# # WindowsError is not defined on unix systems
# try:
#     WindowsError
# except NameError:
#     WindowsError = None

# #
# # Rules
# #
# def find_files(*ext):
#     for root, _, files in os.walk('..'):
#         if 'caer' in root and not '.git' in root:
#             for file in files:
#                 if file.endswith(ext):
#                     fi = root + '\\' + file
#                     CYTHON_SOURCES.append(fi.replace('\\', '/')[3:])

# def process_pyx():
#     flags = ['-3', '--inplace']

#     try:
#         # try the cython in the installed python first (somewhat related to scipy/scipy#2397)
#         from Cython.Compiler.Version import version as cython_version
#     except ImportError:
#         raise OSError('Cython needs to be installed')

#     # Cython 0.29.21 is required for Python 3.9 and there are
#     # other fixes in the 0.29 series that are needed even for earlier
#     # Python versions.
#     # Note: keep in sync with that in buildproject.toml
#     required_cython_version = '0.29.21'

#     if cython_version < required_cython_version:
#         raise RuntimeError(f'Building Caer requires Cython >= {required_cython_version}')
#     # subprocess.check_call(
#     #     [sys.executable, '-m', 'cython'] + flags + ["-o", tofile, fromfile])

#     find_files('.py')
#     # Can only concatenate lists
#     subprocess.check_call(
#         [sys.executable, '-m', 'cythonize'] + flags + CYTHON_SOURCES)



# # def process_tempita_pyx(fromfile, tofile):
# #     import npy_tempita as tempita

# #     assert fromfile.endswith('.pyx.in')
# #     with open(fromfile, "r") as f:
# #         tmpl = f.read()
# #     pyxcontent = tempita.sub(tmpl)
# #     pyxfile = fromfile[:-len('.pyx.in')] + '.pyx'
# #     with open(pyxfile, "w") as f:
# #         f.write(pyxcontent)
# #     process_pyx(pyxfile, tofile)


# # def process_tempita_pyd(fromfile, tofile):
# #     import npy_tempita as tempita

# #     assert fromfile.endswith('.pxd.in')
# #     assert tofile.endswith('.pxd')
# #     with open(fromfile, "r") as f:
# #         tmpl = f.read()
# #     pyxcontent = tempita.sub(tmpl)
# #     with open(tofile, "w") as f:
# #         f.write(pyxcontent)

# # def process_tempita_pxi(fromfile, tofile):
# #     import npy_tempita as tempita

# #     assert fromfile.endswith('.pxi.in')
# #     assert tofile.endswith('.pxi')
# #     with open(fromfile, "r") as f:
# #         tmpl = f.read()
# #     pyxcontent = tempita.sub(tmpl)
# #     with open(tofile, "w") as f:
# #         f.write(pyxcontent)

# # def process_tempita_pxd(fromfile, tofile):
# #     import npy_tempita as tempita

# #     assert fromfile.endswith('.pxd.in')
# #     assert tofile.endswith('.pxd')
# #     with open(fromfile, "r") as f:
# #         tmpl = f.read()
# #     pyxcontent = tempita.sub(tmpl)
# #     with open(tofile, "w") as f:
# #         f.write(pyxcontent)

# # rules = {
# #     # fromext : function, toext
# #     '.pyx' : (process_pyx, '.c'),
# #     '.pyx.in' : (process_tempita_pyx, '.c'),
# #     '.pxi.in' : (process_tempita_pxi, '.pxi'),
# #     '.pxd.in' : (process_tempita_pxd, '.pxd'),
# #     '.pyd.in' : (process_tempita_pyd, '.pyd'),
# #     }

# rules = {
#     # fromext : function, toext
#     '.pyx' : (process_pyx, '.c')
# }

# #
# # Hash db
# #
# def load_hashes(filename):
#     # Return { filename : (sha1 of input, sha1 of output) }
#     if os.path.isfile(filename):
#         hashes = {}
#         with open(filename, 'r') as f:
#             for line in f:
#                 filename, inhash, outhash = line.split()
#                 hashes[filename] = (inhash, outhash)
#     else:
#         hashes = {}
#     return hashes


# def save_hashes(hash_db, filename):
#     with open(filename, 'w') as f:
#         for key, value in sorted(hash_db.items()):
#             f.write("%s %s %s\n" % (key, value[0], value[1]))


# def sha1_of_file(filename):
#     h = hashlib.sha1()
#     with open(filename, "rb") as f:
#         h.update(f.read())
#     return h.hexdigest()


# #
# # Main program
# #

# def normpath(path):
#     path = path.replace(os.sep, '/')
#     if path.startswith('./'):
#         path = path[2:]
#     return path


# def get_hash(frompath, topath):
#     from_hash = sha1_of_file(frompath)
#     to_hash = sha1_of_file(topath) if os.path.exists(topath) else None
#     return (from_hash, to_hash)


# def process(path, fromfile, tofile, processor_function, hash_db):
#     fullfrompath = os.path.join(path, fromfile)
#     fulltopath = os.path.join(path, tofile)
#     current_hash = get_hash(fullfrompath, fulltopath)
#     if current_hash == hash_db.get(normpath(fullfrompath), None):
#         print(f'{fullfrompath} has not changed')
#         return

#     orig_cwd = os.getcwd()
#     try:
#         os.chdir(path)
#         print(f'Processing {fullfrompath}')
#         processor_function(fromfile, tofile)
#     finally:
#         os.chdir(orig_cwd)
#     # changed target file, recompute hash
#     current_hash = get_hash(fullfrompath, fulltopath)
#     # store hash in db
#     hash_db[normpath(fullfrompath)] = current_hash


# def find_process_files(root_dir):
#     hash_db = load_hashes(HASH_FILE)
#     files  = [x for x in os.listdir(root_dir) if not os.path.isdir(x)]
#     # # .pxi or .pxi.in files are most likely dependencies for
#     # # .pyx files, so we need to process them first
#     # files.sort(key=lambda name: (name.endswith('.pxi') or
#     #                              name.endswith('.pxi.in') or
#     #                              name.endswith('.pxd.in')),
#     #            reverse=True)

#     for filename in files:
#         # in_file = os.path.join(root_dir, filename + ".in")
#         for fromext, value in rules.items():
#             if filename.endswith(fromext):
#                 if not value:
#                     break
#                 function, toext = value
#                 if toext == '.c':
#                     with open(os.path.join(root_dir, filename), 'rb') as f:
#                         data = f.read()
#                         m = re.search(br"^\s*#\s*distutils:\s*language\s*=\s*c\+\+\s*$", data, re.I|re.M)
#                         if m:
#                             toext = ".cxx"
#                 fromfile = filename
#                 tofile = filename[:-len(fromext)] + toext
#                 process(root_dir, fromfile, tofile, function, hash_db)
#                 save_hashes(hash_db, HASH_FILE)
#                 break

# def main():
#     try:
#         root_dir = sys.argv[1]
#     except IndexError:
#         root_dir = ROOT_DIR
#     find_process_files(root_dir)


# if __name__ == '__main__':
#     main()