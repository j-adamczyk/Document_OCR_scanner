import cv2 as cv
import numpy as np
import imutils


def get_rect_size(rect):
    return rect[1][0] * rect[1][1]


def prepare_image(src: np.ndarray):
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))
    return src_gray


def cut_sheet_from_image(src: np.ndarray) -> np.ndarray:
    src_gray = prepare_image(src)

    THRESHOLD = 100  # initial threshold
    canny_output = cv.Canny(src_gray, THRESHOLD, THRESHOLD * 2)

    contours = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    min_rects = [cv.minAreaRect(c) for c in contours]
    biggest_rect = max(min_rects, key=lambda e: get_rect_size(e))

    box = cv.boxPoints(biggest_rect)
    box = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)

    green_mask = np.zeros_like(src)
    cv.drawContours(green_mask, [box], 0, [200, 200, 200], -1)

    binary_mask = np.zeros(green_mask.shape, dtype=bool)
    binary_mask[green_mask != 0] = True

    out = src.copy()
    out[binary_mask != True] = 0
    return out


def show_image(image: np.ndarray):
    cv.namedWindow("Contours", cv.WINDOW_NORMAL)
    cv.imshow('Contours', image)
    cv.resizeWindow("Contours", 1280, 720)
    cv.waitKey(0)
