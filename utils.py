import cv2 as cv
import numpy as np


def read_image_from_file(filename: str) -> np.ndarray:
    """
    Reads image from file.

    Parameters
    ----------
    filename: str
        Image filename

    Returns
    -------
    np.ndarray
        Read BGR image.
    """
    return cv.imread(filename, cv.IMREAD_COLOR)


def get_image_corners(image: np.ndarray) -> list:
    """
    Creates tuple of four points (x, y) arranged anticlockwise,
        starting top-left corner.

    Parameters
    ----------
    image : numpy.ndarray
        Original image (np.ndarray).

    Returns
    -------
    list
        List of four points (x, y) arranged anticlockwise,
        starting top-left corner.
    """
    return [(0, 0), (0, image.shape[0] - 1), (image.shape[1] - 1, image.shape[0] - 1), (image.shape[1] - 1, 0)]


def show_image(image: np.ndarray):
    """
    Shows image in a separate window.

    Parameters
    ----------
    image : numpy.ndarray
        Image to show.
    """
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
