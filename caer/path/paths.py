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
from ..jit.annotations import List

_acceptable_video_formats = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
_acceptable_image_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


__all__ = [
    'list_media',
    'list_images',
    'list_videos',
    'mkdir',
    'listdir',
    'is_image',
    'is_video',
    'isfile',
    'cwd',
    'exists',
    'minijoin',
    'get_size',
    'chdir',
    'osname',
    'abspath',
    'dirname'
]


def list_images(DIR, recursive=True, use_fullpath=False, show_size=False, verbose=1) -> List[str]:
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
    images = _get_media_from_dir(DIR=DIR, recursive=recursive, use_fullpath=use_fullpath, show_size=show_size, list_image_files=True, verbose=verbose)

    if images is not None:   
        return images # images is a list


def list_videos(DIR, recursive=True, use_fullpath=False, show_size=False, verbose=1) -> List[str]:
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

    videos = _get_media_from_dir(DIR=DIR, recursive=recursive, use_fullpath=use_fullpath, show_size=show_size, list_video_files=True, verbose=verbose)

    if videos is not None:   
        return videos # videos is a list


def list_media(DIR, recursive=True, use_fullpath=False, show_size=True, verbose=0) -> List[str]:
    r"""
        Lists all media files within a specific directory (and sub-directories if `recursive=True`)
    
    Args:
        DIR (str): Directory to search for media files
        recursive (bool): Indicate whether to search all subdirectories as well (default = False)
        use_fullpath (bool): Include full filepaths in the returned list (default = False)
        show_size (bool): Prints the disk size of the media files (default = False)
    
    Returns:
        media_files (list): List of names (or full filepaths if `use_fullpath=True`) of the media files

    """

    media = _get_media_from_dir(DIR=DIR, recursive=recursive, use_fullpath=use_fullpath, show_size=show_size, list_image_files=True, list_video_files=True, verbose=verbose)

    if media is not None:   
        return media # media is a list


def _get_media_from_dir(DIR, recursive=True, use_fullpath=False, show_size=True,  list_image_files=False, list_video_files=False, verbose=1) -> List[str]:
    r"""
        Lists all media files within a specific directory (and sub-directories if `recursive=True`)
    
    Args:
        DIR (str): Directory to search for media files
        recursive (bool): Indicate whether to search all subdirectories as well (default = False)
        use_fullpath (bool): Include full filepaths in the returned list (default = False)
        show_size (bool): Prints the disk size of the media files (default = False)
    
    Returns:
        media_files (list): List of names (or full filepaths if `use_fullpath=True`) of the media files

    """

    if not exists(DIR):
        raise ValueError('Specified directory does not exist')

    list_media_files = False
    if list_video_files and list_image_files:
        list_media_files = True
    
    video_files = []
    image_files = []
    size_image_list = 0
    size_video_list = 0
    
    if recursive:
        for root, _, files in os.walk(DIR):
            for file in files:
                fullpath = minijoin(root, file).replace('\\', '/')
                decider = _is_extension_acceptable(file)

                if decider == -1:
                    continue

                elif decider == 0: # if image
                    size_image_list += get_size(fullpath, disp_format='mb')
                    if use_fullpath:
                        image_files.append(fullpath)
                    else:
                        image_files.append(file)

                elif decider == 1: # if video
                    size_video_list += get_size(fullpath, disp_format='mb')
                    if use_fullpath:
                        video_files.append(fullpath)
                    else:
                        video_files.append(file)

    else:
        for file in os.listdir(DIR):
            fullpath = minijoin(DIR, file).replace('\\', '/')
            decider = _is_extension_acceptable(file)
                
            if decider == -1:
                continue

            elif decider == 0: # if image
                size_image_list += get_size(fullpath, disp_format='mb')
                if use_fullpath:
                    image_files.append(fullpath)
                else:
                    image_files.append(file)

            elif decider == 1: # if video
                size_video_list += get_size(fullpath, disp_format='mb')
                if use_fullpath:
                    video_files.append(fullpath)
                else:
                    video_files.append(file)

    count_image_list = len(image_files)
    count_video_list = len(video_files)
    
    if count_image_list == 0 and count_video_list == 0:
        print('[ERROR] No media files were found')

    else:
        if list_media_files:
            if verbose != 0:
                tot_count = count_image_list + count_video_list
                print(f'[INFO] {tot_count} files found')
                if show_size:
                    tot_size = size_image_list + size_video_list
                    print(f'[INFO] Total disk size of media files were {tot_size:.2f}Mb ')

            media_files = image_files + video_files
            return media_files

        elif list_image_files:
            if verbose != 0:
                print(f'[INFO] {count_image_list} images found')
                if show_size:
                    print(f'[INFO] Total disk size of media files were {size_image_list:.2f}Mb ')

            return image_files

        elif list_video_files:
            if verbose != 0:
                print(f'[INFO] {count_video_list} videos found')
                if show_size:
                    print(f'[INFO] Total disk size of videos were {size_video_list:.2f}Mb ')

            return video_files
        


def listdir(DIR, recursive=True, use_fullpath=False, show_size=True, verbose=1) -> List[str]:
    r"""
        Lists all files within a specific directory (and sub-directories if `recursive=True`)
    
    Args:
        DIR (str): Directory to search for files
        recursive (bool): Indicate whether to search all subdirectories as well (default = False)
        use_fullpath (bool): Include full filepaths in the returned list (default = False)
        show_size (bool): Prints the disk size of the files (default = False)
    
    Returns:
        files (list): List of names (or full filepaths if `use_fullpath=True`) of the files

    """

    if not exists(DIR):
        raise ValueError('Specified directory does not exist')
    
    if not isinstance(recursive, bool):
        raise ValueError('recursive must be a boolean')

    if not isinstance(use_fullpath, bool):
        raise ValueError('use_fullpath must be a boolean')

    if not isinstance(show_size, bool):
        raise ValueError('show_size must be a boolean')

    dirs = []
    count_files= 0
    size_dirs_list = 0
    
    if recursive:
        for root, _, files in os.walk(DIR):
            for file in files:
                fullpath = minijoin(root, file).replace('\\', '/')
                size_dirs_list += get_size(fullpath, disp_format='mb')
                if use_fullpath:
                    dirs.append(fullpath)
                else:
                    dirs.append(file)

    else:
        for file in os.listdir(DIR):
            fullpath = minijoin(DIR, file).replace('\\', '/')
            size_dirs_list += get_size(fullpath, disp_format='mb')
            if use_fullpath:
                dirs.append(fullpath)
            else:
                dirs.append(file)

    if verbose != 0:
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


def _is_extension_acceptable(path) -> bool:
    """
        0 --> Image
        1 --> Video
    """
    # char_total = len(file)
    # # Finding the last index of '.' to grab the extension
    # try:
    #     idx = file.rindex('.')
    # except ValueError:
    #     return -1
    # file_ext = file[idx:char_total]

    # if file_ext in _acceptable_image_formats:
    #     return 0 
    # elif file_ext in _acceptable_video_formats:
    #     return 1
    # else:
    #     return -1

    if is_image(path):
        return 0
    elif is_video(path):
        return 1 
    else:
        return -1


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

    if os.path.exists(path):
        return True 
    return False


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


def minijoin(*paths) -> str:
    r"""
        Join multiple filepaths together

    """

    return os.path.join(*paths)


def dirname(file) -> str:
    r"""
        Returns the base directory name of `file`

    """

    return os.path.dirname(file)