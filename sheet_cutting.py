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
    print(type(contours))
    contours = grab_contours(contours)
    min_rects = [cv.minAreaRect(c) for c in contours]

    def get_rect_size(rect: np.ndarray):
        # format enforced by opencv
        # RotatedRect:
        # ((center_x,center_y,(width,heigth),angle)
        return rect[1][0] * rect[1][1]

    biggest_rect = max(min_rects, key=lambda e: get_rect_size(e))

    box = cv.boxPoints(biggest_rect)
    box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)

    green_mask = np.zeros_like(src)
    cv.drawContours(image=green_mask, contours=[box], contourIdx=0, color=[200, 200, 200], thickness=-1)

    binary_mask = np.zeros(green_mask.shape, dtype=bool)
    binary_mask[green_mask != 0] = True

    out = src.copy()
    out[binary_mask != True] = 0
    return out
