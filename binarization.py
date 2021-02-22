import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_local

def binarize(img: np.ndarray) -> np.ndarray:
    """
    Function that binarizes an image (adaptively).

    Parameters
    ----------
    img : numpy.ndarray
        input image (floating point format)

    Returns
    -------
    numpy.ndarray
        binary image
    """

    # converting to grayscale if needed
    if len(img.shape) == 3:
        img = rgb2gray(img)

    # getting and applying the threshold
    # block_size and offset values may not be ideal but they seem to work well
    threshold = threshold_local(img,
                         block_size=21, 
                         offset=0.015, 
                         method='gaussian')
    out = img > threshold

    return out
