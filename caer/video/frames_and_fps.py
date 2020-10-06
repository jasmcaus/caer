# Copyright (c) 2020 Jason Dsouza <jasmcaus@gmail.com>
# Protected under the MIT License (see LICENSE)

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