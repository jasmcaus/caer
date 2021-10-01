#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++, Cuda
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|  \_\

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2021 The Caer Authors <http://github.com/jasmcaus>


from .stream import Stream
from ..path import exists

__all__ = [
    'count_frames',
    'get_fps'
]


def count_frames(video_path):
    r"""
        Returns the number of frames in a video at ``video_path``.

    Args:
        video_path (str): Video Filepath
    
    Returns:
        Frame count 

    """
    if video_path is None:
        raise ValueError('Specify a valid video path')

    stream = Stream(video_path)
    frame_count = stream.count_frames()
    stream.release()
    return frame_count


def get_fps(video_path=None):
    r"""
        Returns the FPS in a video at ``video_path``.

    Args:
        video_path (str): Video Filepath
    
    Returns:
        FPS value. 

    """
    if video_path is None and not exists(video_path):
        raise ValueError('Specify a valid video path')

    stream = Stream(video_path)
    fps_count = stream.get_fps()
    stream.release()
    
    return fps_count