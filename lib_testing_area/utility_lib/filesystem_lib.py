# STD LIBRARIES
import json
import pathlib
import random
from PIL import Image, ImageDraw
from .picture_class import Picture
import cv2

# PERSONAL LIBRARIES
TOP_K_EDGE = 1

# ==== Disk read ====
def fixed_choice():
    target_picture_path = pathlib.Path('../../datasets/raw_phishing/comunidadejesusteama.org.br.png')

    return target_picture_path

def random_choice(target_dir):
    pathlist = pathlib.Path(target_dir).glob('**/*.png')
    target_picture_path = random.choice(list(pathlist))

    return target_picture_path

def get_Pictures_from_directory(directory_path, class_name=Picture):
    pathlist = pathlib.Path(directory_path).glob('**/*.png')
    picture_list = []

    for i , path in enumerate(pathlist):
        tmp_Picture = class_name(id=i, path=path)

        # Store hash
        picture_list.append(tmp_Picture)

    return picture_list

# ==== Disk write ====

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
        curr_pic = cv2.imread(str(path))

        if curr_pic is None or curr_pic.shape == [] or curr_pic.shape[0] == 0 or curr_pic.shape[1] == 0 :
            print("Void picture (to delete ?) : " + str(path))
            # path.unlink()

