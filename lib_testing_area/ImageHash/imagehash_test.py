# STD LIBRARIES
import os
import sys
from enum import Enum, auto
from typing import List

import imagehash
from PIL import Image

# PERSONAL LIBRARIES
sys.path.append(os.path.abspath(os.path.pardir))
from utility_lib import filesystem_lib
from utility_lib import picture_class
from utility_lib import execution_handler

# ENUMERATION
class HASH_TYPE(Enum):
    A_HASH = auto()
    P_HASH = auto()
    P_HASH_SIMPLE = auto()
    D_HASH = auto()
    D_HASH_VERTICAL = auto()
    W_HASH = auto()

# CONFIGURATION
HASH_CHOICE = HASH_TYPE.P_HASH
save_picture = False

class Local_Picture(picture_class.Picture):
    '''
    Overwrite of the parent class function to match with ImageHash requirements
    '''

    def compute_distance_ext(self, pic1, pic2):
        distance = abs(pic1.hash - pic2.hash)
        return distance


# ==== Hashing ====
def hash_pictures(picture_list : List[Local_Picture]):

    for i, curr_picture in enumerate(picture_list):
        # Load and Hash picture
        hash_picture(curr_picture)
        if i % 40 == 0 :
            print(f"Picture {i} out of {len(picture_list)}")

    return picture_list

def hash_picture(curr_picture: Local_Picture):
    try:
        if HASH_CHOICE == HASH_TYPE.A_HASH : # Average
            target_hash = imagehash.average_hash(Image.open(curr_picture.path))
        elif HASH_CHOICE == HASH_TYPE.P_HASH : # Perception
            target_hash = imagehash.phash(Image.open(curr_picture.path))
        elif HASH_CHOICE == HASH_TYPE.P_HASH_SIMPLE : # Perception - simple
            target_hash = imagehash.phash_simple(Image.open(curr_picture.path))
        elif HASH_CHOICE == HASH_TYPE.D_HASH : # D
            target_hash = imagehash.dhash(Image.open(curr_picture.path))
        elif HASH_CHOICE == HASH_TYPE.D_HASH_VERTICAL : # D-vertical
            target_hash = imagehash.dhash_vertical(Image.open(curr_picture.path))
        elif HASH_CHOICE == HASH_TYPE.W_HASH : # Wavelet
            target_hash = imagehash.whash(Image.open(curr_picture.path))
        else :
            raise Exception('IMAGEHASH WRAPPER : HASH_CHOICE NOT CORRECT')

        # TO NORMALIZE : https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5
        curr_picture.hash = target_hash
    except Exception as e:
        print("Error during hashing : " + str(e))

    return curr_picture

# ==== Action definition ====

class Image_hash_execution_handler(execution_handler.Execution_handler) :
    def TO_OVERWRITE_prepare_dataset(self):
        print("Hash pictures ... ")
        self.picture_list = hash_pictures(self.picture_list)

    def TO_OVERWRITE_prepare_target_picture(self):
        self.target_picture = hash_picture(self.target_picture)

if __name__ == '__main__':
    target_dir = "../../datasets/raw_phishing/"
    filesystem_lib.clean_folder(target_dir)

    eh = Image_hash_execution_handler(target_dir=target_dir, Local_Picture=Local_Picture)
    # eh.do_random_test()
    eh.do_full_test()
