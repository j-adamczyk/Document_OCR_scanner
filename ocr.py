import numpy as np
import pandas as pd
from pytesseract import image_to_data, Output


def parse_paragraph(text: pd.Series) -> str:
    """Helper function turning pd.series of words into a string"""
    text.dropna(inplace=True)
    text = text.to_list()
    text = " ".join(text)
    return text

def ocr(img: np.ndarray):
    """
    Uses tesseract to recognise words, outputs them by paragraphs.

    Parameters
    ----------
    img : numpy.ndarray
        input image (ubyte!)

    Returns
    -------
    list of dicts representing paragraphs:
        dict: {"text": str, "dims": (x, y, width, height)}

        example output:
        [
            { "text": "Some text in a paragraph",
            "dims": (222, 1831, 1619, 118) },

            { "text": "another paragraph",
            "dims": (222, 1950, 288, 32) },
        ]

    """
    # ocr
    results = image_to_data(img, 
                lang='eng+pol', 
                output_type=Output.DATAFRAME)

    output_text = []
    output_dims = []
    last_i=0

    # iterare over paragraphs
    mask = results['par_num']>results['par_num'].shift(1)

    for i in results[mask].index:
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]

        output_dims.append( (x, y, w, h) )
        output_text.append( parse_paragraph(results['text'][last_i:i]) )

        last_i=i

    output_text.append( parse_paragraph(results['text'][last_i:]) )
    

    # remove paragraphs with without text
    output = []
    for text, dims in zip(output_text[1:], output_dims):
        if text.strip() != "":
            output.append({"text":text, "dims":dims})

    return output
