import cv2 as cv
import pytesseract

from ocr import ocr

# path to tesseract if not in PATH
# pytesseract.pytesseract.tesseract_cmd = '<path to tesseract>'

def ocr_demo(img):
	"""
	Just to show how it works

	Parameters
    ----------
    img : numpy.ndarray
        PREPROCESSED image (ubyte, grayscale)

    """

	output = ocr(img)

	img_color = cv.merge([img,img,img])

	for i, par in enumerate(output):
	    x = par["dims"][0]
	    y = par["dims"][1]
	    w = par["dims"][2]
	    h = par["dims"][3]

	    text = par["text"]

	    print(i)
	    print("dims: ", (x, y, w, h))
	    print("text: ", text)
	    print()

	    cv.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
	    cv.putText(img_color, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1.2, (127, 0, 127), 3)



	img_color = cv.resize(img_color, (img.shape[0]//3, img.shape[1]//3))
	cv.imshow('ocr test', img_color)
	cv.waitKey(0)
