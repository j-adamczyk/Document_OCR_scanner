import numpy as np
from skimage import data, transform
import warnings

def perspective_transform(img: np.ndarray, input_points: list, output_shape: tuple) -> np.ndarray:
	"""
	Applies 4 point perspective transform to an image and returns it as a separate image.

	Parameters
    ----------
    img : numpy.ndarray
    	Original image (np.ndarray)
    input_points : list of tuples
        List of four points arranged anticlockwise starting top-left corner
    output_shape: tuple
        Desired shape of transformed image (height, width)

    Returns
    -------
    numpy.ndarray
        Transformed image

	"""

	#transformation coordinates
	dst = np.array(input_points)
	src = np.array([[0, 0], [0, output_shape[0]], [output_shape[1], output_shape[0]], [output_shape[1], 0]])

	#checking if input_points are correct 
	if not dst.shape == (4,2):
		warnings.warn(f"Shape of input_points expected to be (4,2), got {dst.shape} instead.\n") 
	elif not (dst[0,1]<dst[1,1] and dst[1,0]<dst[2,0] and dst[2,1]>dst[3,1] and dst[3,0]>dst[0,0]):
		warnings.warn("Suspicious arrangement of input_points, expected anticlockwise starting top-left corner.", )

	#transformation
	tform = transform.ProjectiveTransform()
	tform.estimate(src, dst)
	warped_img = transform.warp(img, tform, output_shape=output_shape)

	return warped_img



