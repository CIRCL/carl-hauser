from PIL import Image, ImageFont, ImageDraw
from .picture_class import Picture
from typing import List
from .json_class import remove_target_picture_from_matches

import pathlib
import configuration
import logging

OFFSETS = 10


class Printer():
    def __init__(self, conf: configuration.Default_configuration, offsets=OFFSETS):
        self.offsets = offsets
        self.logger = logging.getLogger(__name__)

        # self.conf = conf

    # @staticmethod
    def text_and_outline(self, draw, x, y, text, font_size):
        fillcolor = "red"
        shadowcolor = "black"
        outline_size = 1

        fontPath = pathlib.Path("./../utility_lib/fonts/OpenSans-Bold.ttf")
        sans16 = ImageFont.truetype(str(fontPath.resolve()), font_size)

        draw.text((x - outline_size, y - outline_size), text, font=sans16, fill=shadowcolor)
        draw.text((x + outline_size, y - outline_size), text, font=sans16, fill=shadowcolor)
        draw.text((x - outline_size, y + outline_size), text, font=sans16, fill=shadowcolor)
        draw.text((x + outline_size, y + outline_size), text, font=sans16, fill=shadowcolor)
        draw.text((x, y), text, fillcolor, font=sans16)

    # @staticmethod
    def save_picture_top_matches(self, sorted_picture_list: List[Picture], target_picture: Picture, file_name='test.png'):
        image_path_list = []
        image_name_list = []

        # Preprocess to remove target picture from matches
        offset = remove_target_picture_from_matches(sorted_picture_list, target_picture)

        image_path_list.append(str(target_picture.path))
        image_name_list.append("ORIGINAL IMAGE")

        for i in range(0, 3):
            image_path_list.append(str(sorted_picture_list[i + offset].path))
            image_name_list.append("BEST MATCH #" + str(i + offset) + " d=" + str(sorted_picture_list[i + offset].distance))

        images = map(Image.open, image_path_list)
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        images = map(Image.open, image_path_list)  # Flushed between now on the first variable assignement for unknown reason

        draw = ImageDraw.Draw(new_im)

        x_offset = 0
        for i, im in enumerate(images):
            new_im.paste(im, (x_offset, 0))
            tmp_title = image_name_list[i] + " " + str(pathlib.Path(image_path_list[i]).name)

            self.logger.debug(f"ADDING picture : {tmp_title}")

            self.text_and_outline(draw, x_offset, 10, tmp_title, total_width // 120)
            x_offset += im.size[0]

        new_im.save(file_name)

    def print_title(self, img, title):
        width, height, _ = img.shape

        draw = ImageDraw.Draw(img)
        self.text_and_outline(draw, self.offsets, self.offsets, title, width // 120)

        return img
