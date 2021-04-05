import cv2 as cv
import numpy as np

def morphological_closing(denoised_img: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    denoised_img: numpy.ndarray
        denoised image for which the closing should be applied

    Returns
    -------
    numpy.ndarray
        image closed morphologically
    """
    # parameters such as kernel size and num of iterations might require change in the future
    kernel = np.ones((3,3))
    closing = cv.morphologyEx(denoised_img, cv.MORPH_CLOSE, kernel,iterations=3)
    return cv.bitwise_or(denoised_img, closing)

