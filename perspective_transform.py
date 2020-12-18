import numpy as np
from skimage import data, transform

def perspective_transform(img: np.ndarray, input_points: list, output_shape: tuple = None) -> np.ndarray:
	"""
	Applies 4 point perspective transform to an image and returns it as a separate image.

       	Args:
            img (numpy.ndarray): Original image
            input_points (list): List of four points arranged anticlockwise starting top-left corner
            output_shape (tuple): Desired shape of transformed image (height, width), 
        		if not passed, it gets eyeballed based on resolution
        		and A4-like sheet proportions 

        Returns:
        	warped_img (numpy.ndarray): Transformed image
	"""

	dst = np.array(input_points)

	#checking if input_points are correct 
	if not dst.shape == (4,2):
		raise ValueError(f"Shape of input_points expected to be (4,2), got {dst.shape} instead")

	elif not (dst[0,1]<dst[1,1] and dst[1,0]<dst[2,0] and dst[2,1]>dst[3,1] and dst[3,0]>dst[0,0]):
		print("Perspective_transform: suspicious arrangement of input_points", end = " - ")
		print("expected anticlockwise starting top-left corner")

	#eyeballing output_shape if not passed
	if not output_shape:
		from math import dist

		h = max(dist(input_points[0], input_points[1]), dist(input_points[2], input_points[3]))
		w = max(dist(input_points[1], input_points[2]), dist(input_points[3], input_points[0]))

		base = min((h/1.41+w)//2, 2000)
		output_shape = (int(base*1.41), int(base))

		print(f"Perspective_transform: output_shape automatically set to {output_shape}")

	src = np.array([[0, 0], [0, output_shape[0]], [output_shape[1], output_shape[0]], [output_shape[1], 0]])

	#transformation
	tform = transform.ProjectiveTransform()
	tform.estimate(src, dst)
	warped_img = transform.warp(img, tform, output_shape=output_shape)

	return warped_img

