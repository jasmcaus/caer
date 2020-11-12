# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

#pylint:disable=redefined-outer-name

import os

_acceptable_video_formats = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
_acceptable_image_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def list_images(DIR, include_subdirs=True, use_fullpath=False, show_size=False, verbose=1):
    """
        Lists all image files within a specific directory (and sub-directories if `include_subdirs=True`)
    
    Parameters
    ----------
        DIR : str
            Directory to search for image files
        include_subdirs : bool
            Indicate whether to search all subdirectories as well (default = False)
        use_fullpath : bool
            Include full filepaths in the returned list (default = False)
        show_size : bool
            Prints the disk size of the image files (default = False)
    
    Returns
    ----------
        image_files : list
            List of names (or full filepaths if `use_fullpath=True`) of the image files
    """
    images = _get_media_from_dir(DIR=DIR, include_subdirs=include_subdirs, use_fullpath=use_fullpath, show_size=show_size, list_image_files=True, verbose=verbose)

    if images is not None:   
        return images # images is a list


def list_videos(DIR, include_subdirs=True, use_fullpath=False, show_size=False, verbose=1):
    """
        Lists all video files within a specific directory (and sub-directories if `include_subdirs=True`)
    
    Parameters
    ----------
        DIR : str
            Directory to search for video files
        include_subdirs : bool
            Indicate whether to search all subdirectories as well (default = False)
        use_fullpath : bool
            Include full filepaths in the returned list (default = False)
        show_size : bool
            Prints the disk size of the video files (default = False)
    
    Returns
    ----------
        video_files : list
            List of names (or full filepaths if `use_fullpath=True`) of the video files
    """

    videos = _get_media_from_dir(DIR=DIR, include_subdirs=include_subdirs, use_fullpath=use_fullpath, show_size=show_size, list_video_files=True, verbose=verbose)

    if videos is not None:   
        return videos # videos is a list


def list_media(DIR, include_subdirs=True, use_fullpath=False, show_size=True, verbose=0):
    """
        Lists all media files within a specific directory (and sub-directories if `include_subdirs=True`)
    
    Parameters
    ----------
        DIR : str
            Directory to search for media files
        include_subdirs : bool
            Indicate whether to search all subdirectories as well (default = False)
        use_fullpath : bool
            Include full filepaths in the returned list (default = False)
        show_size : bool
            Prints the disk size of the media files (default = False)
    
    Returns
    ----------
        media_files : list
            List of names (or full filepaths if `use_fullpath=True`) of the media files
    """

    media = _get_media_from_dir(DIR=DIR, include_subdirs=include_subdirs, use_fullpath=use_fullpath, show_size=show_size, list_image_files=True, list_video_files=True, verbose=verbose)

    if media is not None:   
        return media # media is a list


def _get_media_from_dir(DIR, include_subdirs=True, use_fullpath=False, show_size=True,  list_image_files=False, list_video_files=False, verbose=1):
    """
        Lists all media files within a specific directory (and sub-directories if `include_subdirs=True`)
    
    Parameters
    ----------
        DIR : str
            Directory to search for media files
        include_subdirs : bool
            Indicate whether to search all subdirectories as well (default = False)
        use_fullpath : bool
            Include full filepaths in the returned list (default = False)
        show_size : bool
            Prints the disk size of the media files (default = False)
    
    Returns
    ----------
        media_files : list
            List of names (or full filepaths if `use_fullpath=True`) of the media files
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
    
    if include_subdirs:
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
                    print('[INFO] Total disk size of media files were {:.2f}Mb '.format(tot_size))

            media_files = image_files + video_files
            return media_files

        elif list_image_files:
            if verbose != 0:
                print(f'[INFO] {count_image_list} images found')
                if show_size:
                    print('[INFO] Total disk size of media files were {:.2f}Mb '.format(size_image_list))

            return image_files

        elif list_video_files:
            if verbose != 0:
                print(f'[INFO] {count_video_list} videos found')
                if show_size:
                    print('[INFO] Total disk size of videos were {:.2f}Mb '.format(size_video_list))

            return video_files
        


def listdir(DIR, include_subdirs=True, use_fullpath=False, show_size=True, verbose=1):
    """
        Lists all files within a specific directory (and sub-directories if `include_subdirs=True`)
    
    Parameters
    ----------
        DIR : str
            Directory to search for files
        include_subdirs : bool
            Indicate whether to search all subdirectories as well (default = False)
        use_fullpath : bool
            Include full filepaths in the returned list (default = False)
        show_size : bool
            Prints the disk size of the files (default = False)
    
    Returns
    ----------
        files : list
            List of names (or full filepaths if `use_fullpath=True`) of the files
    """

    if not exists(DIR):
        raise ValueError('Specified directory does not exist')
    
    if not isinstance(include_subdirs, bool):
        raise ValueError('include_subdirs must be a boolean')

    if not isinstance(use_fullpath, bool):
        raise ValueError('use_fullpath must be a boolean')

    if not isinstance(show_size, bool):
        raise ValueError('show_size must be a boolean')

    dirs = []
    count_files= 0
    size_dirs_list = 0
    
    if include_subdirs:
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
            print('[INFO] Total disk size of files were {:.2f}Mb '.format(size_dirs_list))

    return dirs


def is_image(path):
    """
        Checks if a given path is that of a valid image file
            
    Parameters
    ----------
        path : str
            Filepath to check
    
    Returns
    ----------
        True; if `path` is a valid image filepath
        False; otherwise
    """

    if not isinstance(path, str):
        raise ValueError('path must be a string')

    if path.endswith(_acceptable_image_formats):
        return True 

    return False


def is_video(path):
    """
        Checks if a given path is that of a valid image file
            
    Parameters
    ----------
        path : str
            Filepath to check
    
    Returns
    ----------
        True; if `path` is a valid image filepath
        False; otherwise
    """
    if not isinstance(path, str):
        raise ValueError('path must be a string')

    if path.endswith(_acceptable_video_formats):
        return True 

    return False


def _is_extension_acceptable(path):
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


def osname():
    return os.name


def cwd():
    """
        Returns the filepath to the current working directory
    """
    return os.getcwd()


def exists(path):
    """
        Checks if a given filepath is valid
            
    Parameters
    ----------
        path : str
            Filepath to check
    
    Returns
    ----------
        True; if `path` is a valid filepath
        False; otherwise
    """

    if not isinstance(path, str):
        raise ValueError('Filepath must be a string')

    if os.path.exists(path):
        return True 
    return False


def isdir(DIR):
    """
        Checks if a given filepath is that of a directory
            
    Parameters
    ----------
        path : str
            Filepath to check
    
    Returns
    ----------
        True; if `path` is a valid directory
        False; otherwise
    """
    return os.path.isdir(DIR)


def mkdir(path):
    """
        Creates a directory at `path`
    """
    os.mkdir(path)


def abspath(file_name):
    """
        Returns the absolute path of `file_name`
    """
    return os.path.abspath(file_name)


def chdir(path):
    """
        Checks into directory `path`
    """
    if not isinstance(path, str):
        raise ValueError('Specify a valid path')

    os.chdir(path)


def get_size(file, disp_format='bytes'):
    """
        Returns the size of `file` in bytes/kb/mb/gb/tb
            
    Parameters
    ----------
        file : str
            Filepath to check
        disp_format : str
            Size format (bytes/kb/mb/gb/tb)
    
    Returns
    ----------
        size : str
            File size in bytes/kb/mb/gb/tb
    """

    if not isinstance(disp_format, str):
        raise ValueError('display format must be a string')

    if disp_format.lower() not in ['bytes', 'kb', 'mb', 'gb', 'tb']:
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


def minijoin(*paths):
    """
        Join multiple filepaths together
    """

    return os.path.join(*paths)


def dirname(file):
    """
        Returns the base directory name of `file`
    """
    
    return os.path.dirname(file)


__all__ = [
    'list_media',
    'list_images',
    'list_videos',
    'mkdir',
    'listdir',
    'is_image',
    'is_video'
    'cwd',
    'exists',
    'minijoin',
    'get_size',
    'chdir',
    'osname',
    'abspath',
    'dirname'
]