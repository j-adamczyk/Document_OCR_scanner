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
   
    # estimating right block_size value
    block_size = 1 +  2 * int((img.shape[0] + img.shape[1]) / 250)
    block_size = min(block_size, 35)
    block_size = max(block_size, 15)

    # getting theshold mask and applying it
    threshold = threshold_local(img,
                         block_size = 21, 
                         offset = 0.015, 
                         method = 'gaussian')

    out = img > threshold

    return out
