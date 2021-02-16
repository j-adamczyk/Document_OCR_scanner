import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.restoration import denoise_bilateral
from skimage.transform import probabilistic_hough_line, rotate


def align_text(image: np.ndarray) -> np.ndarray:
    """
    Function which rotates sheet to get aligned, parallel text

    Parameters
    ----------
    image : numpy.ndarray
        photo with skewed text, it must be RGB (not gray or RGBA !!!)

    Returns
    -------
    numpy.ndarray
        image with aligned, parallel text

    """

    # rgb -> gray
    draft = rgb2gray(image)

    # filtering image to preserve lines better
    filtered_img = denoise_bilateral(draft)

    # detecting edges and applying probabilistic line Hough transform
    edges = canny(filtered_img)
    lines = probabilistic_hough_line(edges)

    # every hough line needs to be transformed to slope value and then to the angle (like in linear functions)
    slopes = np.array([(y2 - y1) / (x2 - x1) if (x2 - x1) else 0 for (x1, y1), (x2, y2) in lines])
    deg_angles = np.degrees(np.arctan(slopes))

    # getting the most common degree value
    hist, angles = np.histogram(deg_angles, bins=180)
    rotation_angle: float = angles[np.argmax(hist)]

    # computing correct rotate angle
    if rotation_angle < -45:                 # -50 -> 40 (we don't want text to lose horizontal/vertical orientation)
        rotation_angle = 90 - abs(rotation_angle)
    elif rotation_angle > 45:                # 50 -> -40 (same as before)
        rotation_angle = -(90 - rotation_angle)

    # rotating and returning image
    img = rotate(image, angle=rotation_angle)
    return img
