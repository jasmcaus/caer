# Author: Jason Dsouza
# Github: http://www.github.com/jasmcaus

""" Provides visualization-specific functions """

def hex_to_rgb(x):
    """
    Turns a color hex representation into a tuple representation.
    """
    return tuple([int(x[i:i + 2], 16) for i in (0, 2, 4)])


def draw_rectangle(draw, coordinates, color, width=1, fill=30):
    """Draw a rectangle with an optional width."""
    # Add alphas to the color so we have a small overlay over the object.
    fill = color + (fill,)
    outline = color + (255,)

    # Pillow doesn't support width in rectangles, so we must emulate it with a
    # loop.
    for i in range(width):
        coords = [
            coordinates[0] - i,
            coordinates[1] - i,
            coordinates[2] + i,
            coordinates[3] + i,
        ]

        # Fill must be drawn only for the first rectangle, or the alphas will
        # add up.
        if i == 0:
            draw.rectangle(coords, fill=fill, outline=outline)
        else:
            draw.rectangle(coords, outline=outline)