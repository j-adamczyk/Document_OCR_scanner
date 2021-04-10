import numpy as np
from skimage.filters import unsharp_mask


def unsharp(image: np.ndarray) -> np.ndarray:
    """
    Function which unsharp image

    Parameters
    ----------
    image : numpy.ndarray
        binarizated photo

    Returns
    -------
    numpy.ndarray
        unsharped image

    """

    # using unsharp masking to make image cleaner, radius and amount values can be changed in the future
    result = unsharp_mask(image, radius=5, amount=2)

    return result
