# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

from .videostream import VideoStream

def count_frames(video_path=None):
    if video_path is None:
        raise ValueError('[ERROR] Specify a valid video path')

    stream = VideoStream(video_path)
    frame_count = stream.count_frames()
    stream.release()
    return frame_count