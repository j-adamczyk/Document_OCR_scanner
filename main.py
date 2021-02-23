import argparse

import utils
from perspective_transform import perspective_transform
from sheet_cutting import get_corners_of_sheet
from text_align import align_text

parser = argparse.ArgumentParser()
parser.add_argument("image_filename")


def main():
    args = parser.parse_args()
    image = utils.read_image_from_file(args.image_filename)

    corners = get_corners_of_sheet(image.copy())

    image = perspective_transform(image.copy(), corners, image.shape)
    image = align_text(image.copy())
    utils.show_image(image, "transformed")


if __name__ == '__main__':
    main()
