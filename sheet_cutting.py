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


def convert_clockwise_to_anticlockwise(points: list) -> list:
    """
    Parameters
    ----------
    points
        list of 4 points arranged clockwise
    Returns
    -------
        list
            points arranged anti-clockwise
    """
    tmp = points[1].copy()
    points[1] = points[3]
    points[3] = tmp
    return points


def get_corners_of_sheet(src: np.ndarray) -> list:
    """
       Get corners of sheet of paper from the photo.
       Parameters
       ----------
       src : numpy.ndarray
           source photo BGR
       Returns
       -------
       list
           list of corners of detected document arranged anit-clockwise
       """
    src_gray = gray_and_blur_image(src)

    THRESHOLD = 100  # initial threshold
    canny_output = cv.Canny(src_gray, THRESHOLD, THRESHOLD * 2)

    # contours are arranged clockwise according to documentation
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
    return convert_clockwise_to_anticlockwise(box)
