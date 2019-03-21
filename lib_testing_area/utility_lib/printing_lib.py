from PIL import Image, ImageFont, ImageDraw
from .picture_class import Picture
import pathlib
from typing import List
from .json_class import remove_target_picture_from_matches

def text_and_outline(draw, x, y, text, font_size):
    fillcolor = "red"
    shadowcolor = "black"
    outline_size = 1

    fontPath = "./fonts/OpenSans-Bold.ttf"
    sans16 = ImageFont.truetype(fontPath, font_size)

    draw.text((x - outline_size, y - outline_size), text, font=sans16, fill=shadowcolor)
    draw.text((x + outline_size, y - outline_size), text, font=sans16,fill=shadowcolor)
    draw.text((x - outline_size, y + outline_size), text, font=sans16,fill=shadowcolor)
    draw.text((x + outline_size, y + outline_size), text, font=sans16,fill=shadowcolor)
    draw.text((x, y), text, fillcolor, font=sans16 )


def save_picture_top_matches(sorted_picture_list : List[Picture], target_picture : Picture, file_name='test.png') :
    image_path_list = []
    image_name_list = []

    # Preprocess to remove target picture from matches
    offset = remove_target_picture_from_matches(sorted_picture_list,target_picture)

    image_path_list.append(str(target_picture.path))
    image_name_list.append("ORIGINAL IMAGE")

    for i in range(0,3):
        image_path_list.append(str(sorted_picture_list[i+offset].path))
        image_name_list.append("BEST MATCH #" + str(i+offset) + " d=" + str(sorted_picture_list[i+offset].distance))

    images = map(Image.open, image_path_list)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    images = map(Image.open, image_path_list) # Droped between now on the previous assignement for unknown reason

    draw = ImageDraw.Draw(new_im)

    x_offset = 0
    for i, im in enumerate(images):
        new_im.paste(im, (x_offset,0))
        tmp_title = image_name_list[i] + " " + str(pathlib.Path(image_path_list[i]).name)

        print(f"ADDING picture : {tmp_title}")

        text_and_outline(draw,x_offset,10,tmp_title, total_width//120)
        x_offset += im.size[0]

    new_im.save(file_name)
