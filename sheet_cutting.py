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

def grab_contours(cnts):
    '''
    Utility method copied from https://github.com/jrosebr1/imutils
    Parameters
    ----------
    cnts
        contours tuple
    Returns
        the actual contours array
    -------

    '''
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
                         "otherwise OpenCV changed their cv2.findContours return "
                         "signature yet again. Refer to OpenCV's documentation "
                         "in that case"))

    # return the actual contours array
    return cnts


def convert_clockwise_to_anticlockwise(points: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    points: numpy.ndarray
        list of 4 points arranged clockwise
    Returns
    -------
    numpy.ndarray
        points arranged anti-clockwise
    """
    points[[1, 3]] = points[[3, 1]]
    return points


def get_corners_of_sheet(src: np.ndarray) -> np.ndarray:
    """
       Get corners of sheet of paper from the photo.
       Parameters
       ----------
       src : numpy.ndarray
           source photo BGR
       Returns
       -------
       np.ndarray
           2D array of corners of detected document arranged anit-clockwise
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
        # ((center_x, center_y), (width, height), angle)
        return rect[1][0] * rect[1][1]

    biggest_rect = max(min_rects, key=lambda e: get_rect_size(e))
    box = cv.boxPoints(biggest_rect)
    return convert_clockwise_to_anticlockwise(box)
