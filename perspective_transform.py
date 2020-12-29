import numpy as np
from skimage import data, transform

def perspective_transform(img: np.ndarray, input_points: list, output_shape: tuple) -> np.ndarray:
	"""
	Applies 4 point perspective transform to an image and returns it as a separate image.

       	:param img:
       		Original image (np.ndarray)
        :param input_points: 
        	List of four points arranged anticlockwise starting top-left corner
        :param output_shape: 
        	Desired shape of transformed image (height, width)

        :raises:
        	ValueError: if input_points have wrong shape 

        :returns: 
        	Transformed image (np.ndarray)
	"""

	#transformation coordinates
	dst = np.array(input_points)
	src = np.array([[0, 0], [0, output_shape[0]], [output_shape[1], output_shape[0]], [output_shape[1], 0]])

	#checking if input_points are correct 
	if not dst.shape == (4,2):
		raise ValueError(f"Shape of input_points expected to be (4,2), got {dst.shape} instead")
		
	if not (dst[0,1]<dst[1,1] and dst[1,0]<dst[2,0] and dst[2,1]>dst[3,1] and dst[3,0]>dst[0,0]):
		print("Perspective_transform: suspicious arrangement of input_points", end = " - ")
		print("expected anticlockwise starting top-left corner")

	#transformation
	tform = transform.ProjectiveTransform()
	tform.estimate(src, dst)
	warped_img = transform.warp(img, tform, output_shape=output_shape)

	return warped_img


#Simple example
'''
from skimage.data import checkerboard
from skimage.viewer import ImageViewer

sample_img = checkerboard()
input_points = [(10, 10), (10, 150), (140, 120), (100, 10)]
img = perspective_transform(sample_img, input_points, (400,300))

ImageViewer(img).show()
'''

