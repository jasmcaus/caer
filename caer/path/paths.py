#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


#pylint:disable=redefined-outer-name

import os
from ..annotations import List, Union

_acceptable_video_formats = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
_acceptable_image_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

__all__ = [
    'list_images',
    'list_videos',
    'listdir',
    'is_image',
    'is_video',
    'isfile',
    'mkdir',
    'cwd',
    'exists',
    'join',
    'get_size',
    'chdir',
    'osname',
    'abspath',
    'dirname'
]

def list_images(DIR, recursive=True, use_fullpath=False, show_size=False, verbose=True) -> List[str]:
    r"""
        Lists all image files within a specific directory (and sub-directories if `recursive=True`)
    
    Args:
        DIR (str): Directory to search for image files
        recursive (bool): Indicate whether to search all subdirectories as well (default = False)
        use_fullpath (bool): Include full filepaths in the returned list (default = False)
        show_size (bool): Prints the disk size of the image files (default = False)
    
    Returns:
        image_files (list): List of names (or full filepaths if `use_fullpath=True`) of the image files

    """
    return listdir(DIR=DIR, recursive=recursive, use_fullpath=use_fullpath, ext = _acceptable_image_formats, show_size=show_size, verbose=verbose)


def list_videos(DIR, recursive=True, use_fullpath=False, show_size=False, verbose=True) -> List[str]:
    r"""
        Lists all video files within a specific directory (and sub-directories if `recursive=True`)
    
    Args:
        DIR (str): Directory to search for video files
        recursive (bool): Indicate whether to search all subdirectories as well (default = False)
        use_fullpath (bool): Include full filepaths in the returned list (default = False)
        show_size (bool): Prints the disk size of the video files (default = False)
    
    Returns:
        video_files (list): List of names (or full filepaths if `use_fullpath=True`) of the video files

    """

    return listdir(DIR=DIR, recursive=recursive, use_fullpath=use_fullpath, ext = _acceptable_video_formats, show_size=show_size, verbose=verbose)


def listdir(
        DIR : str,
        recursive : bool = False, 
        use_fullpath: bool = False, 
        ext : Union[str, List[str]] = None, 
        show_size : bool = False, 
        verbose : bool = True
    ) -> List[str]:
    r"""
        Lists all files within a specific directory (and sub-directories if `recursive=True`).
        This can be filtered for certain extensions (by populating `ext`)
    
    Args:
        DIR (str): Directory to search for files
        recursive (bool): Indicate whether to search all subdirectories as well (default = False)
        use_fullpath (bool): Include full filepaths in the returned list (default = False)
        ext (str, list(str), tuple(str)): Filter by extension names.
        show_size (bool): Prints the disk size of the files (default = False)
        verbose(bool): Print info
    
    Returns:
        files (list): List of names (or full filepaths if `use_fullpath=True`) of the files

    """

    if not exists(DIR):
        raise ValueError('Specified directory does not exist')
    
    if not isinstance(recursive, bool):
        raise ValueError('recursive must be a boolean')

    if not isinstance(use_fullpath, bool):
        raise ValueError('use_fullpath must be a boolean')

    if ext is not None:
        if not isinstance(ext, (str, list, tuple)):
            raise TypeError("`ext` must either be of type `str`, or tuple/list of `str`")

        if isinstance(ext, (list, tuple)):
            for i in ext:
                if not isinstance(i, str):
                    raise TypeError("`ext` must be a homogenous list of `str`")

    if not isinstance(show_size, bool):
        raise ValueError('show_size must be a boolean')
    
    if not isinstance(verbose, bool):
        raise ValueError('verbose must be a boolean')

    dirs : list = []
    count_files : int = 0
    size_dirs_list : int = 0
    
    if recursive:
        for root, _, files in os.walk(DIR):
            for file in files:
                if ext is not None and not file.endswith(ext):
                    continue
                fullpath = join(root, file).replace('\\', '/')
                size_dirs_list += get_size(fullpath, disp_format='mb')
                if use_fullpath:
                    dirs.append(fullpath)
                else:
                    dirs.append(file)

    else:
        for file in os.listdir(DIR):
            if ext is not None and not file.endswith(ext):
                continue
            fullpath = join(DIR, file).replace('\\', '/')
            size_dirs_list += get_size(fullpath, disp_format='mb')
            if use_fullpath:
                dirs.append(fullpath)
            else:
                dirs.append(file)

    if verbose:
        count_files = len(dirs)
        if count_files == 1:
            print(f'[INFO] {count_files} file found')
        else:
            print(f'[INFO] {count_files} files found')

        if show_size:
            print(f'[INFO] Total disk size of files were {size_dirs_list:.2f}Mb ')

    return dirs


def is_image(path) -> bool:
    r"""
        Checks if a given path is that of a valid image file
            
    Args:
        path (str): Filepath to check
    
    Returns:
        True; if `path` is a valid image filepath
        False; otherwise

    """

    if not isinstance(path, str):
        raise ValueError('path must be a string')

    if path.endswith(_acceptable_image_formats):
        return True 

    return False


def is_video(path) -> bool:
    r"""
        Checks if a given path is that of a valid image file
            
    Args:
        path (str): Filepath to check
    
    Returns:
        True; if `path` is a valid image filepath
        False; otherwise

    """
    if not isinstance(path, str):
        raise ValueError('path must be a string')

    if path.endswith(_acceptable_video_formats):
        return True 

    return False


def osname() -> str:
    return os.name


def cwd() -> str:
    r"""
        Returns the filepath to the current working directory
    """
    return os.getcwd()


def exists(path) -> bool:
    r"""
        Checks if a given filepath is valid
            
    Args:
        path (str): Filepath to check
    
    Returns:
        True; if `path` is a valid filepath
        False; otherwise

    """

    if not isinstance(path, str):
        raise ValueError('Filepath must be a string')

    return os.path.exists(path)


def isfile(path) -> bool:
    r"""
        Checks if a given filepath is a valid path
            
    Args:
        path (str): Filepath to check
    
    Returns:
        True; if `path` is a valid filepath
        False; otherwise

    """
    return os.path.isfile(path)


def isdir(path) -> bool:
    r"""
        Checks if a given filepath is that of a directory
            
    Args:
        path (str): Filepath to check
    
    Returns:
        True; if `path` is a valid directory
        False; otherwise

    """
    return os.path.isdir(path)


def mkdir(path) -> None:
    r"""
        Creates a directory at `path`

    """
    os.mkdir(path)


def abspath(file_name) -> str:
    r"""
        Returns the absolute path of `file_name`

    """
    return os.path.abspath(file_name)


def chdir(path) -> None:
    r"""
        Checks into directory `path`

    """
    if not isinstance(path, str):
        raise ValueError('Specify a valid path')

    os.chdir(path)


def get_size(file, disp_format='bytes') -> float:
    r"""
        Returns: the size of `file` in bytes/kb/mb/gb/tb
            
    Args:
        file (str): Filepath to check
        disp_format (str): Size format (bytes/kb/mb/gb/tb)
    
    Returns:
        size (str): File size in bytes/kb/mb/gb/tb

    """

    if not isinstance(disp_format, str):
        raise ValueError('display format must be a string')
    
    disp_format = disp_format.lower()

    if disp_format not in ['bytes', 'kb', 'mb', 'gb', 'tb']:
        raise ValueError('display format needs to be either bytes/kb/mb/gb/tb')

    size = os.path.getsize(file)

    if disp_format == 'bytes':
        return size 

    if disp_format == 'kb':
        return size * 1e-3

    if disp_format == 'mb':
        return size * 1e-6

    if disp_format == 'gb':
        return size * 1e-9

    if disp_format == 'tb':
        return size * 1e-12


def join(*paths) -> str:
    r"""
        Join multiple filepaths together

    """

    return os.path.join(*paths)


def dirname(file) -> str:
    r"""
        Returns the base directory name of `file`

    """

    return os.path.dirname(file)