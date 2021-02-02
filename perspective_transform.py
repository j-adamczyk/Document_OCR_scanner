import numpy as np
from skimage import data, transform
import warnings

def perspective_transform(
            img: np.ndarray, 
            input_points: list, 
            output_shape: tuple) -> np.ndarray:
    """
    Applies 4 point perspective transform to an image
    and returns it as a separate image.

    Parameters
    ----------
    img : numpy.ndarray
        Original image (np.ndarray).
    input_points : list of tuples
        List of four points (x, y) arranged anticlockwise,
        starting top-left corner.
    output_shape: tuple
        Desired shape of transformed image (height, width).

    Returns
    -------
    numpy.ndarray
        Transformed image.

    Raises
    ------
    ValueError
        If shape of input_points is not (4,2).

    """

    # checking if input_points are in correct shape
    input_shape = np.array(input_points).shape
    if not input_shape == (4,2):
        raise ValueError("Shape of input_points expected to be (4,2) " +
                           f"got {input_shape} instead.") 

    upper_left_src = (0, 0)
    lower_left_src = (0, output_shape[0])
    lower_right_src = (output_shape[1], output_shape[0])
    upper_right_src = (output_shape[1], 0)

    upper_left_dst = input_points[0]
    lower_left_dst = input_points[1]
    lower_right_dst = input_points[2]
    upper_right_dst = input_points[3]

    # checking if input_points are arranged in a correct way
    if not (upper_left_dst[1] < lower_left_dst[1] 
            and lower_left_dst[0] < lower_right_dst[0] 
            and lower_right_dst[1] > upper_right_dst[1] 
            and upper_right_dst[0] > upper_left_dst[0]):
        warnings.warn("Suspicious arrangement of input_points," +
                    "expected anticlockwise starting top-left corner.")

    # transformation coordinates 
    dst = np.array(input_points)
    src = np.array([upper_left_src, 
                    lower_left_src, 
                    lower_right_src, 
                    upper_right_src])

    # transformation 
    tform = transform.ProjectiveTransform()
    tform.estimate(src, dst)
    warped_img = transform.warp(img, tform, output_shape=output_shape)

    return warped_img
