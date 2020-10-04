# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

""" Provides visualization-specific functions """

def hex_to_rgb(x):
    """
    Turns a color hex representation into a tuple representation.
    """
    return tuple([int(x[i:i + 2], 16) for i in (0, 2, 4)])

