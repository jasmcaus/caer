#    _____           ______  _____ 
#  / ____/    /\    |  ____ |  __ \
# | |        /  \   | |__   | |__) | Caer - Modern Computer Vision
# | |       / /\ \  |  __|  |  _  /  Languages: Python, C, C++
# | |___   / ____ \ | |____ | | \ \  http://github.com/jasmcaus/caer
#  \_____\/_/    \_ \______ |_|__\_

# Licensed under the MIT License <http://opensource.org/licenses/MIT>
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>


from .videostream import VideoStream


__all__ = [
    'count_frames',
    'get_fps'
]


def count_frames(video_path=None):
    if video_path is None:
        raise ValueError('Specify a valid video path')

    stream = VideoStream(video_path)
    frame_count = stream.count_frames()
    stream.release()
    return frame_count


def get_fps(video_path=None):
    if video_path is None:
        raise ValueError('Specify a valid video path')

    stream = VideoStream(video_path)
    fps_count = stream.get_fps()
    stream.release()
    return fps_count