import numpy as np
import skimage.filters
from skimage.restoration import denoise_bilateral


def denoise(image: np.ndarray) -> np.ndarray:
    """
    Function which filters image to remove noise and make letters more continuous

    Parameters
    ----------
    image : numpy.ndarray
        binarizated photo

    Returns
    -------
    numpy.ndarray
        denoised image

    """

    # median filtering to remove useless points or small blurs
    med_img = skimage.filters.median(image)

    # bilateral filtering to make text more smooth - it's better to OCR
    # sigma_color is 0.7 because it gives better continuity of letters than 0,
    # it doesn't bold text too much like value=1 does (experimentally determined)
    filtered_img = denoise_bilateral(med_img, sigma_color=0.7)

    return filtered_img
