# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

# Importing the necessary packages
import os

_acceptable_video_formats = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
_acceptable_image_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def list_images(DIR, include_subdirs=True, use_fullpath=False, get_size=True):
    """
        Lists all image files within a specific directory (and sub-directories if `include_subdirs=True`)
        DIR -> Directory to search for image files
        :paraminclude_subdirs: --> Boolean to indicate whether to search all subdirectories as well
        use_fullpath --> Boolean that specifies whether to include full filepaths in the returned list
        :return image_files: --> List of names (or full filepaths if `use_fullpath=True`) of the image files
    """
    images = _get_media_from_dir(DIR=DIR, include_subdirs=include_subdirs, use_fullpath=use_fullpath, get_size=get_size, list_image_files=True)
    return images # images is a list

def list_videos(DIR, include_subdirs=True, use_fullpath=False, get_size=True):
    """
        Lists all video files within a specific directory (and sub-directories if `include_subdirs=True`)
        DIR -> Directory to search for video files
        :paraminclude_subdirs: --> Boolean to indicate whether to search all subdirectories as well
        use_fullpath --> Boolean that specifies whether to include full filepaths in the returned list
        :return video_files: --> List of names (or full filepaths if `use_fullpath=True`) of the video files
    """
    videos = _get_media_from_dir(DIR=DIR, include_subdirs=include_subdirs, use_fullpath=use_fullpath, get_size=get_size, list_video_files=True)
    return videos # videos is a list


def list_media(DIR, include_subdirs=True, use_fullpath=False, get_size=True):
    """
        Lists all media files within a specific directory (and sub-directories if `include_subdirs=True`)
        DIR -> Directory to search for media files
        :paraminclude_subdirs: --> Boolean to indicate whether to search all subdirectories as well
        use_fullpath --> Boolean that specifies whether to include full filepaths in the returned list
        :return media_files: --> List of names (or full filepaths if `use_fullpath=True`) of the media files
    """
    media = _get_media_from_dir(DIR=DIR, include_subdirs=include_subdirs, use_fullpath=use_fullpath, get_size=get_size, list_image_files=True, list_video_files=True)
    return media # media is a list


def _get_media_from_dir(DIR, include_subdirs=True, use_fullpath=False, get_size=True,  list_image_files=False, list_video_files=False):
    """
        Lists all video files within a specific directory (and sub-directories if `include_subdirs=True`)
        DIR -> Directory to search for video files
        :paraminclude_subdirs: --> Boolean to indicate whether to search all subdirectories as well
        use_fullpath --> Boolean that specifies whether to include full filepaths in the returned list
        :return video_list: --> List of names (or full filepaths if `use_fullpath=True`) of the video files
    """
    if not os.path.exists(DIR):
        raise ValueError('[ERROR] Specified directory does not exist')

    list_media_files = False
    if list_video_files and list_image_files:
        list_media_files = True
    
    video_files = []
    image_files = []
    count_image_list = 0
    count_video_list = 0
    size_image_list = 0
    size_video_list = 0
    
    if include_subdirs:
        for root, _, files in os.walk(DIR):
            for file in files:
                fullpath = os.path.join(root,file)
                decider = is_extension_acceptable(file)

                if decider == -1:
                    continue

                elif decider == 0: # if image
                    size_image_list += os.stat(fullpath).st_size/(1024*1024)
                    if use_fullpath:
                        image_files.append(fullpath)
                    else:
                        image_files.append(file)
                    count_image_list += 1

                elif decider == 1: # if video
                    size_video_list += os.stat(fullpath).st_size/(1024*1024)
                    if use_fullpath:
                        video_files.append(fullpath)
                    else:
                        video_files.append(file)
                    count_video_list += 1
    else:
        for file in os.listdir(DIR):
            fullpath = os.path.join(DIR,file)
            decider = is_extension_acceptable(file)
                
            if decider == -1:
                continue

            elif decider == 0: # if image
                size_image_list += os.stat(fullpath).st_size/(1024*1024)
                if use_fullpath:
                    image_files.append(fullpath)
                else:
                    image_files.append(file)
                count_image_list += 1

            elif decider == 1: # if video
                size_video_list += os.stat(fullpath).st_size/(1024*1024)
                if use_fullpath:
                    video_files.append(fullpath)
                else:
                    video_files.append(file)
                count_video_list += 1


    if list_media_files:
        tot_count = count_image_list + count_video_list
        print(f'[INFO] {tot_count} files found')
        if get_size:
            tot_size = size_image_list + size_video_list
            print('[INFO] Total disk size of media files were {:.2f}Mb '.format(tot_size))
        media_files = image_files + video_files
        return media_files

    elif list_image_files:
        print(f'[INFO] {count_image_list} images found')
        if get_size:
            print('[INFO] Total disk size of media files were {:.2f}Mb '.format(size_image_list))
        return image_files

    elif list_video_files:
        print(f'[INFO] {count_video_list} videos found')
        if get_size:
            print('[INFO] Total disk size of videos were {:.2f}Mb '.format(size_video_list))
        return video_files


def is_extension_acceptable(file):
    """
        0 --> Image
        1 --> Video
    """
    char_total = len(file)
    # Finding the last index of '.' to grab the extension
    try:
        idx = file.rindex('.')
    except ValueError:
        return -1
    file_ext = file[idx:char_total]

    if file_ext in _acceptable_image_formats:
        return 0 
    elif file_ext in _acceptable_video_formats:
        return 1
    else:
        return -1

def listdir(DIR, include_subdirs=False):
    if not os.path.exists(DIR):
        raise ValueError('[ERROR] Specified directory does not exist')
    
    count = 0

    if include_subdirs:
        for _,_,files in os.walk(DIR):
            for file in files:
                print(file)
                count += 1
        if count == 1:
            print(f'[INFO] {count} file found')
        else:
            print(f'[INFO] {count} files found')

    else:
        for file in os.listdir(DIR):
            print(file)
            count += 1
        if count == 1:
            print(f'[INFO] {count} file found')
        else:
            print(f'[INFO] {count} files found')

