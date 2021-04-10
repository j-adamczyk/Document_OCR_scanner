# python built-ins
import argparse

# project modules
import utils
from binarization import binarize
from denoising import denoise
from perspective_transform import perspective_transform
from rescaling import rescale
from sheet_cutting import get_corners_of_sheet
from text_align import align_text
from unsharping import unsharp


def get_parsed_args() -> argparse.Namespace:
    """
        Function to get namespace of possible arguments
    Returns
    -------
        argparse.Namespace
            namespace with parsed args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("image_filename")
    parser.add_argument("--denoise", action='store_true')
    parser.add_argument("--unsharp", action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parsed_args()
    image = utils.read_image_from_file(args.image_filename)

    corners = get_corners_of_sheet(image)

    image = perspective_transform(image, corners, image.shape)
    image = align_text(image)
    image = rescale(image)
    image = binarize(image)

    if args.denoise:
        image = denoise(image)
    if args.unsharp:
        image = unsharp(image)
    if not args.denoise and not args.unsharp:
        image = utils.convert_to_float64(image)

    utils.show_image(image, "transformed")
