# python built-ins
import argparse

# project modules
import utils
from perspective_transform import perspective_transform
from sheet_cutting import get_corners_of_sheet
from text_align import align_text


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
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parsed_args()
    image = utils.read_image_from_file(args.image_filename)

    corners = get_corners_of_sheet(image)

    image = perspective_transform(image, corners, image.shape)
    image = align_text(image)
    utils.show_image(image, "transformed")
