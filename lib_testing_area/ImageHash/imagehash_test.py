
import imagehash
import pathlib
import random
import time
import operator
from PIL import Image, ImageFont, ImageDraw
from scipy import stats
from typing import List

import json

choices_list = ["a","p","p-simple", "d", "d-vertical", "w"]
HASH_CHOICE = "w"

json_to_export = {}
hash_list = []
save_picture = False


class Picture() :
    def __init__(self, id, shape="image", path=None, hash=None):
        self.id = id
        self.shape = shape
        self.path = path
        self.matched = False
        self.sorted_matching_picture_list = []
        # Hashing related attributes
        self.hash = hash
        self.distance = None

    def to_node_json_object(self):
        tmp_obj = {}
        tmp_obj["id"] = self.id
        tmp_obj["shape"] = self.shape
        tmp_obj["image"] = self.path.name
        return tmp_obj

    def compute_distance(self, target_hash):
        self.distance = abs(self.hash - target_hash)
        return self.distance


# ==== Disk access ====
def fixed_choice():
    target_picture_path = pathlib.Path('../../datasets/raw_phishing/comunidadejesusteama.org.br.png')

    return target_picture_path

def random_choice(target_dir):
    pathlist = pathlib.Path(target_dir).glob('**/*.png')
    target_picture_path = random.choice(list(pathlist))

    return target_picture_path

def get_Pictures_from_directory(directory_path):
    pathlist = pathlib.Path(directory_path).glob('**/*.png')
    picture_list = []

    for i , path in enumerate(pathlist):
        tmp_Picture = Picture(id=i, path=path)

        # Store hash
        picture_list.append(tmp_Picture)

    return picture_list

# ==== Hashing ====
def hash_pictures(picture_list : List[Picture]):

    for curr_picture in picture_list:
        # Load and Hash picture
        curr_picture = hash_picture(curr_picture)

    return picture_list

def hash_picture(curr_picture: Picture):
    try:
        if HASH_CHOICE == choices_list[0] : # Average
            target_hash = imagehash.average_hash(Image.open(curr_picture.path))
        elif HASH_CHOICE == choices_list [1] : # Perception
            target_hash = imagehash.phash(Image.open(curr_picture.path))
        elif HASH_CHOICE == choices_list [2] : # Perception - simple
            target_hash = imagehash.phash_simple(Image.open(curr_picture.path))
        elif HASH_CHOICE == choices_list [3] : # D
            target_hash = imagehash.dhash(Image.open(curr_picture.path))
        elif HASH_CHOICE == choices_list [4] : # D-vertical
            target_hash = imagehash.dhash_vertical(Image.open(curr_picture.path))
        elif HASH_CHOICE == choices_list [5] : # Wavelet
            target_hash = imagehash.whash(Image.open(curr_picture.path))

        curr_picture.hash = target_hash
    except Exception as e:
        print("Error during hashing : " + str(e))

    return curr_picture

# ==== Checking ====
def find_closest(picture_list : List[Picture], target_picture : Picture):
    min = None
    min_object = None

    for curr_picture in picture_list :
        if not are_same_picture(target_picture,curr_picture) and (min is None or min > abs(curr_picture.hash-target_picture.hash)):
            min = abs(curr_picture.hash-target_picture.hash) # TODO : Hamming .. ?
            min_object = curr_picture

    print("original picture : \t" + str(target_picture.path))
    print("min found : \t" + str(min_object.path) + " with " + str(min))


def get_top(picture_list : List[Picture], target_picture : Picture):
    for curr_picture in picture_list :
        curr_picture.compute_distance(target_picture.hash)

    sorted_picture_list = sorted(picture_list, key=operator.attrgetter('distance'))
    print(sorted_picture_list)

    return sorted_picture_list

def print_list(hash_list):
    THREESHOLD = 5
    for i in hash_list[0:THREESHOLD]:
        print(str(i.path) + " : " + str(i.distance))

def are_same_picture(pic1 : Picture, pic2 : Picture):
    # TODO : Except on SHA1 hash ?
    return pic1.path == pic2.path

def remove_target_picture_from_matches(sorted_picture_list : List[Picture], target_picture : Picture):
    offset = 0
    if are_same_picture(sorted_picture_list[0], target_picture) :
        # If first picture is the original picture we skip.
        print("Removed first choice : " + sorted_picture_list[0].path.name)
        offset += 1

    return offset

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


def json_add_nodes(picture_list : List[Picture]) :
    nodes_list = []

    # Add all nodes
    for curr_picture in picture_list :
        nodes_list.append(curr_picture.to_node_json_object())

    json_to_export["nodes"] = nodes_list

TOP_K_EDGE = 1

def json_add_top_matches(sorted_picture_list : List[Picture], target_picture : Picture) :
    # Preprocess to remove target picture from matches
    offset = remove_target_picture_from_matches(sorted_picture_list,target_picture)

    # Get current list of matches
    edges_list = json_to_export.get("edges", [])

    # Add all edges with labels
    for i in range(0,TOP_K_EDGE) :
        tmp_obj = {}
        tmp_obj["from"] = target_picture.id
        tmp_obj["to"] = sorted_picture_list[i+offset].id
        tmp_obj["label"] = "rank " + str(i) + "(" + str(sorted_picture_list[i+offset].distance) + ")"
        edges_list.append(tmp_obj)

    # Store in JSON variable
    json_to_export["edges"] = edges_list
    
def json_export(json_to_export, file_name='test.json'):
    with open(pathlib.Path(file_name), 'w') as outfile:
        json.dump(json_to_export,  outfile)

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


def random_test(target_dir):
    print("Pick a random picture ... ")
    picture_list = []
    target_picture_path = random_choice(target_dir)
    print("Target picture : " + str(target_picture_path))

    print("Load pictures ... ")
    picture_list = get_Pictures_from_directory(target_dir)

    print("Hash pictures ... ")
    start_time = time.time()
    picture_list = hash_pictures(picture_list)
    elapsed = time.time() - start_time
    print(f"Elapsed hashing time : {elapsed} sec for {len(picture_list)} items ({elapsed/len(picture_list)} per item)")

    print("Hash target picture ... ")
    target_picture = Picture(id=None, path=target_picture_path)
    target_picture = hash_picture(target_picture)

    print("Find closest picture from target picture ... ")
    find_closest(picture_list, target_picture) # TODO : To remove ? Not useful ?

    print("Extract top K images ... ")
    sorted_picture_list = get_top(hash_list, target_picture)

    if save_picture : save_picture_top_matches(sorted_picture_list, target_picture)

def full_test(target_dir):
    pathlist = pathlib.Path(target_dir).glob('**/*.png')
    output_dir = "./RESULTS/"

    print("Load pictures ... ")
    picture_list = get_Pictures_from_directory(target_dir)

    print("Prepare Json file with initial nodes")
    json_add_nodes(picture_list)

    start_time = time.time()
    print(f"Hash pictures from repository {target_dir}... ")
    picture_list = hash_pictures(picture_list)
    elapsed = time.time() - start_time
    print(f"Elapsed hashing time : {elapsed} sec for {len(picture_list)} items ({elapsed / len(picture_list)} per item)")

    start_FULL_time = time.time()
    list_time = []
    for i, curr_target_picture in enumerate(picture_list):
        start_time = time.time()
        print(f"PICTURE {i} picked as target ... ")
        print("Target picture : " + str(curr_target_picture.path))

        print("Find closest picture from target picture ... ")
        find_closest(picture_list, curr_target_picture)

        print("Extract top K images ... ")
        sorted_picture_list = get_top(picture_list, curr_target_picture)

        elapsed = time.time() - start_time
        print(f"Elapsed hashing time : {elapsed} sec")
        list_time.append(elapsed)

        print("Export as picture ... ")
        if save_picture : save_picture_top_matches(sorted_picture_list, curr_target_picture, file_name= output_dir + curr_target_picture.path.name + "_RESULT.png")
        print("Save result for final Json ... ")
        json_add_top_matches(sorted_picture_list, curr_target_picture)

    elapsed_FULL_TIME = time.time() - start_FULL_time
    print(f"Elapsed hashing time : {elapsed_FULL_TIME} sec for {len(picture_list)} items ({elapsed_FULL_TIME / len(picture_list)} per item)")

    print("Export json ... ")
    json_export(json_to_export, "test.json")

    stats_result = stats.describe(list_time)

ROUND_DECIMAL = 5

def print_stats(stats):
    tmp_str = ""
    tmp_str += "nobs : " + str( getattr(stats_result, "nobs")) + "s "
    tmp_str += "min time : " + str(round(getattr(stats_result, "minmax")[0],ROUND_DECIMAL)) + "s "
    tmp_str += "max time : " + str(round(getattr(stats_result, "minmax")[1],ROUND_DECIMAL)) + "s "
    tmp_str += "mean :" + str(getattr(stats_result, "mean")) + "s "
    tmp_str += "variance : " + str(getattr(stats_result, "variance")) + "s "
    tmp_str += "skewness : " + str(getattr(stats_result, "skewness") ) + "s "
    tmp_str += "kurtosis : " + str(getattr(stats_result, "kurtosis") )
    print(tmp_str)
def clean_folder(target_dir):
    '''
    Remove 0-bytes size files
    :param target_dir:
    :return:
    '''

    pathlist = pathlib.Path(target_dir).glob('**/*.png')
    for i, path in enumerate(pathlist):
        if path.stat().st_size == 0 :
            path.unlink()

if __name__ == '__main__':
    target_dir = "../../datasets/raw_phishing/"
    clean_folder(target_dir)
    # random_test(target_dir)
    full_test(target_dir)
