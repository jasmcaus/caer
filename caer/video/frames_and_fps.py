#
#  _____ _____ _____ ____
# |     | ___ | ___  | __|  Caer - Modern Computer Vision
# |     |     |      | \    version 3.9.1
# |_____|     | ____ |  \   http://github.com/jasmcaus/caer

# Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# SPDX-License-Identifier: MIT
# Copyright (c) 2020 The Caer Authors <http://github.com/jasmcaus>.
# 

from .videostream import VideoStream


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


__all__ = [
    'count_frames',
    'get_fps'
]