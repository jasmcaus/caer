# Copyright 2020 The Caer Authors. All Rights Reserved.
#
# Licensed under the MIT License (see LICENSE);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at <https://opensource.org/licenses/MIT>
#
# ==============================================================================

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