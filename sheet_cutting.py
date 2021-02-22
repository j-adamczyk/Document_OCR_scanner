import cv2 as cv
import numpy as np


def gray_and_blur_image(src: np.ndarray):
    """
       convert src to grayscale and blur it
       Parameters
       ----------
       src : numpy.ndarray
           source photo BGR
       Returns
       -------
       numpy.ndarray
           blurred grayscale image
       """
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))
    return src_gray


def grab_contours(contours: tuple) -> tuple:
    """
       Grabs contours using imutils library
       Parameters
       ----------
       contours : tuple
           tuple of contours found by opencv
       Returns
       -------
       tuple
       contours
       """
    import imutils
    return imutils.grab_contours(contours)


def get_up_left_and_bot_right_corners(binary_array_2d: np.ndarray) -> tuple:
    """
       Calculates leftmost upmost and righmost bottommost points with True value
       Parameters
       ----------
       binary_array_2d : numpy.ndarray
           source binary mask in 2D
       Returns
       -------
       numpy.ndarray
           tuple of points: leftmost upmost and righmost bottommost corners of 2D array of booleans
       """
    positive_cells_coords = np.argwhere(binary_array_2d)
    x_set = set(map(lambda x: x[1], positive_cells_coords))
    y_set = set(map(lambda x: x[0], positive_cells_coords))
    x_min = min(x_set)
    x_max = max(x_set)
    y_min = min(y_set)
    y_max = max(y_set)
    return (x_min, y_min), (x_max, y_max)


def flatten_3d_binary_mask(image: np.ndarray) -> np.array:
    """
       Flattens 3D binary mask to 2D with binary OR
       Parameters
       ----------
       image : numpy.ndarray
           source binary mask in 3D
       Returns
       -------
       numpy.ndarray
           flattened binary mask
       """
    flat = np.zeros(shape=(image.shape[0], image.shape[1]))
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            flat[x][y] = image[x][y][0] or image[x][y][1] or image[x][y][2]
    return flat


def cut_sheet_from_image(src: np.ndarray) -> np.ndarray:
    """
       Cuts sheet of paper from the photo
       Parameters
       ----------
       src : numpy.ndarray
           source photo BGR
       Returns
       -------
       numpy.ndarray
           cut sheet BGR
       """
    src_gray = gray_and_blur_image(src)

    THRESHOLD = 100  # initial threshold
    canny_output = cv.Canny(src_gray, THRESHOLD, THRESHOLD * 2)

    contours = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)
    min_rects = [cv.minAreaRect(c) for c in contours]

    def get_rect_size(rect: np.ndarray):
        # format enforced by opencv
        # RotatedRect:
        # ((center_x,center_y),(width,height),angle)
        return rect[1][0] * rect[1][1]

    biggest_rect = max(min_rects, key=lambda e: get_rect_size(e))

    box = cv.boxPoints(biggest_rect)
    box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)

    green_mask = np.zeros_like(src)
    cv.drawContours(image=green_mask, contours=[box], contourIdx=0, color=[200, 200, 200], thickness=-1)

    binary_mask = np.zeros(green_mask.shape, dtype=bool)
    binary_mask[green_mask != 0] = True
    flat_binary_mask = flatten_3d_binary_mask(binary_mask)

    up_left, bottom_right = get_up_left_and_bot_right_corners(flat_binary_mask)
    out = src.copy()
    out[flat_binary_mask != True] = 0
    return out[up_left[1]:bottom_right[1], up_left[0]:bottom_right[0]]
