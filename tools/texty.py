
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Hello world app',
    ext_modules=cythonize(['../caer/configs.py', '../caer/images.py', '../caer/opencv.py', '../caer/preprocess.py', '../caer/setup.py', '../caer/time.py', '../caer/utilities.py', '../caer/visualizations.py', '../caer/_base.py', '../caer/_checks.py', '../caer/_meta.py', '../caer/_sklearn_utils.py', '../caer/_split.py', '../caer/_spmatrix.py', '../caer/__init__.py', '../caer/data/__init__.py', '../caer/path/paths.py', '../caer/path/__init__.py', '../caer/preprocessing/mean_subtraction.py', '../caer/preprocessing/patch_preprocess.py', '../caer/preprocessing/_patches.py', '../caer/preprocessing/__init__.py', '../caer/utils/exceptions.py', '../caer/utils/validators.py', '../caer/utils/__init__.py', '../caer/video/extract_frames.py', '../caer/video/filevideostream.py', '../caer/video/frames_and_fps.py', '../caer/video/gpufilevideostream.py', '../caer/video/livevideostream.py', '../caer/video/videostream.py', '../caer/video/__init__.py']),
    zip_safe=False,
)
