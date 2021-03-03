import io
import numpy as np
from PIL import Image


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
        rescaled image (to constant 300 DPI)

    """

    # saving image with 300 DPI to temporary file in RAM using BytesIO and PIL save function with DPI scaling
    buffer = io.BytesIO()
    pil_img = Image.fromarray(image)
    pil_img.save(buffer, format="png", dpi=(300, 300))

    # setting start of bytes stream
    buffer.seek(0)

    # getting rescaled image and changing format to np.ndarray
    rescaled_img = Image.open(buffer)
    reformatted_img = np.array(rescaled_img)

    return reformatted_img
