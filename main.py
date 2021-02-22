import argparse

import utils
from perspective_transform import perspective_transform
from sheet_cutting import cut_sheet_from_image
from text_align import align_text

parser = argparse.ArgumentParser()
parser.add_argument("image_filename")


def main():
    args = parser.parse_args()
    image = utils.read_image_from_file(args.image_filename)
    image = cut_sheet_from_image(image)
    image = perspective_transform(image.copy(), utils.get_image_corners(image), (600, 600))
    image = align_text(image.copy())
    utils.show_image(image)


if __name__ == '__main__':
    main()
