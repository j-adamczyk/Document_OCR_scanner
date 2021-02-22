import matplotlib.pyplot as matplot
import numpy as np
import os
from skimage import img_as_ubyte
from skimage.color import rgb2gray
import skimage.io as scikit


def rescale(image: np.ndarray) -> np.ndarray:
    """
    Function which rescales image to improve OCR process

    Parameters
    ----------
    image : numpy.ndarray
        photo to rescale

    Returns
    -------
    numpy.ndarray
        rescaled image (to constant 300 DPI), as gray (more info below)

    """

    # name of temporary file
    filename = "rescaled_img.jpg"

    # the only way to change DPI I've found is to save image using matplotlib.pyplot.imsave() function
    # so, I'm saving with 300 DPI, then loading again and deleting temporary file
    matplot.imsave(filename, image, dpi=300)
    rescaled_img = scikit.imread(filename)
    os.remove(filename)

    # converting to gray because saved tmp file has black text on yellow background
    gray_img = rgb2gray(rescaled_img)

    return gray_img


def test_run(filename: str):
    """
    Function which loads image and calls rescaling function written above.
    This function is not intented to use in Document OCR Scanner project,
    it's just for local tests and can be removed. The right method to use is above.

    Parameters
    ----------
    filename : string
        name of file with photo to test

    """

    # reading image with scikit image function
    img = scikit.imread(filename)

    # rescaling process
    rescaled_img = rescale(img)

    # saving image to file with scikit image function
    scikit.imsave(f"{filename.split('.')[0]}_rescaled.jpg", img_as_ubyte(rescaled_img))

