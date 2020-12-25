from skimage.io import imread, imsave
from skimage.filters import gaussian, threshold_otsu
from skimage.color import rgb2gray, rgba2rgb
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, rotate
from skimage import img_as_ubyte
import numpy as np


def align_text(image: np.ndarray) -> np.ndarray:
    """
        Rotate sheet to get aligned, parallel text

        :param image:
            loaded image as numpy.ndarray type

        :return:
            rotated image as numpy.ndarray type
    """

    # checking if image is gray, rgba or rgb
    draft = image
    if len(draft.shape) > 2:
        if draft.shape[2] == 4:         # converting from rgba to rgb
            draft = rgba2rgb(draft)
        if draft.shape[2] == 3:         # converting from rgb to gray
            draft = rgb2gray(draft)

    # Otsu's method - returns threshold to determine text block
    thresh: float = threshold_otsu(draft)

    # creating binary image where (True = text) and converting it to gaussian blur
    binary_img: np.ndarray = draft > thresh
    gaussian_blur: np.ndarray = gaussian(binary_img, 3)

    # detecting edges and adopting probabilistic line Hough transform
    edges: np.ndarray = canny(gaussian_blur)
    lines = probabilistic_hough_line(edges)

    # every hough line needs to be transformed to slope value and then to the angle (like in linear functions)
    slopes = [(y2 - y1) / (x2 - x1) if (x2 - x1) else 0 for (x1, y1), (x2, y2) in lines]
    deg_angles = [np.degrees(np.arctan(x)) for x in slopes]

    # getting most common degree value
    histogram = np.histogram(deg_angles, bins=180)
    rotation_angle: float = histogram[1][np.argmax(histogram[0])]

    # computing correct rotate angle
    if rotation_angle < -45:                 # -50 -> 40 (we don't want text to lose horizontal/vertical orientation)
        rotation_angle = 90 - abs(rotation_angle)
    elif rotation_angle > 45:                # 50 -> -40 (same as before)
        rotation_angle = -(90 - rotation_angle)

    # rotating and returning image
    img = rotate(image, angle=rotation_angle)
    return img


# example
file_name: str = "skewed.jpg"
img_file: np.ndarray = imread(file_name)
rotated_image = align_text(img_file)
imsave("rotated_" + file_name, img_as_ubyte(rotated_image))
