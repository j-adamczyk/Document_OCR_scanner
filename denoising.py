import numpy as np
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, median
from skimage.io import imsave, imread
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
    med_img = median(image)

    # bilateral filtering to make text more smooth - it's better to OCR
    # sigma_color is 0.7 because it gives better continuity of letters than 0,
    # it doesn't bold text too much like value=1 does (experimentally determined)
    filtered_img = denoise_bilateral(med_img, sigma_color=0.7)

    return filtered_img


def test_run(filename: str):
    """
    Function which prepares image by binarizing and runs denoising function written above.
    This function is not intented to use in Document OCR Scanner project, it's just for
    local tests and can be removed. The right method to use is above.

    Parameters
    ----------
    filename : string
        name of file with photo to test

    """

    # reading image with scikit image function
    img = imread(filename)

    # grayscaling image - recommended for Otsu's method
    gray_img = rgb2gray(img)

    # binarisation process
    threshold = threshold_otsu(gray_img)
    bin_img = gray_img > threshold

    # denoising process
    result_img = denoise(bin_img)

    # saving image to file with scikit image function
    imsave(f"{filename.split('.')[0]}_res_col070.jpg", img_as_ubyte(result_img))


