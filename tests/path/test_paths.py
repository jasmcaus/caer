#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>

import os 
import caer

# Note: 
# os.path.join(os.path.dirname(os.getcwd()), 'media') is the correct convention (for accessing tests/media)
# However, pytest runs differently and doesn't call os.getcwd() as we would expect (i.e it runs it from the root dir 'caer')
# Hence, we add an additional 'tests' for Pytest to run correctly

# PATH_TO_MEDIA_FILES = os.path.join(os.getcwd(), 'media')
PATH_TO_MEDIA_FILES = os.path.join(os.getcwd(), 'tests', 'path', 'media')


def test_list_images():
    DIR = PATH_TO_MEDIA_FILES

    images_list = caer.path.list_images(DIR, recursive=True, use_fullpath=True)

    assert images_list is not None 
    assert len(images_list) == 3 # There are 3 images & 3 videos in tests/media

    for i in images_list:
        assert os.path.exists(i) 


def test_list_videos():
    DIR = PATH_TO_MEDIA_FILES

    videos_list = caer.path.list_videos(DIR, recursive=True, use_fullpath=True)

    assert videos_list is not None 
    assert len(videos_list) == 3 #There are 3 images & 3 videos in tests/media

    for i in videos_list:
        assert os.path.exists(i) 


def test_listdir():
    DIR = PATH_TO_MEDIA_FILES

    dir_list = caer.path.listdir(DIR, recursive=True, use_fullpath=True)

    assert dir_list is not None 
    assert len(dir_list) == 7 # There are 3 images, 3 videos and 1 README.md in tests/media

    for i in dir_list:
        assert os.path.exists(i) 


def test_is_image():
    IMAGE = os.path.join(PATH_TO_MEDIA_FILES, 'silvestri-matteo-6-C0VRsagUw-unsplash.jpg')

    assert caer.path.is_image(IMAGE)


def test_is_video():
    VIDEO = os.path.join(PATH_TO_MEDIA_FILES, 'Bird - 46026.mp4')

    assert caer.path.is_video(VIDEO)


def test_isfile():
    FILE = os.path.join(PATH_TO_MEDIA_FILES, 'silvestri-matteo-6-C0VRsagUw-unsplash.jpg')

    assert caer.path.isfile(FILE) == os.path.isfile(FILE)


def test_isdir():
    DIR = PATH_TO_MEDIA_FILES

    assert caer.path.isdir(DIR) == os.path.isdir(DIR)


def test_get_size():
    FILE = os.path.join(PATH_TO_MEDIA_FILES, 'Nature', 'Bird - 46026.mp4')

    assert caer.path.get_size(FILE, 'bytes') == os.path.getsize(FILE)
    assert caer.path.get_size(FILE, 'kb') == (os.path.getsize(FILE) * 1e-3)
    assert caer.path.get_size(FILE, 'mb') == (os.path.getsize(FILE) * 1e-6)
    assert caer.path.get_size(FILE, 'gb') == (os.path.getsize(FILE) * 1e-9)
    assert caer.path.get_size(FILE, 'tb') == (os.path.getsize(FILE) * 1e-12)


def test_osname():
    s = caer.path.osname()

    assert s == os.name


def test_cwd():
    s = caer.path.cwd()

    assert s == os.getcwd()


def test_abspath():
    s = caer.path.abspath(__file__)

    assert s == os.path.abspath(__file__)
    

def test_dirname():
    s = caer.path.dirname(__file__)

    assert s == os.path.dirname(__file__)