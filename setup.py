import setuptools

long_description = """
A Computer Vision library in Python with powerful image processing operations, including support for Deep Learning models built using the Keras framework

This repository is actively being maintained. If there are any issues, kindly open a thread in the 'Issues' pane on the official Github repository. 
"""

setuptools.setup(
    name="caer",
    version="0.0.5",
    author="Jason Dsouza",
    author_email="jasmcaus@gmail.com",
    description="A Computer Vision library in Python with powerful image processing operations, including support for Deep Learning models built using the Keras framework",
    long_description=long_description,
    url="https://github.com/jasmcaus/caer",
    packages=setuptools.find_packages(),
    license='MIT',
    install_requires=['tensorflow', 'numpy', 'opencv-contrib-python', 'os'],
    keywords=['computer vision', 'deep learning', 'image processing', 'opencv', 'matplotlib'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)