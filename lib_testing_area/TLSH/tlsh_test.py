# STD LIBRARIES
import os

import sys
import tlsh
from enum import Enum, auto
from typing import List

# PERSONAL LIBRARIES
sys.path.append(os.path.abspath(os.path.pardir))
from utility_lib import filesystem_lib, picture_class, execution_handler

# ENUMERATION
class HASH_TYPE(Enum):
    NORMAL = auto()
    NO_LENGTH = auto()

# CONFIGURATION
HASH_CHOICE = HASH_TYPE.NORMAL
save_picture = False

class Local_Picture(picture_class.Picture):
    '''
    Overwrite of the parent class function to match with ImageHash requirements
    '''

    def compute_distance_ext(self, pic1, pic2):
        dist = None
        if HASH_CHOICE == HASH_TYPE.NORMAL:
            dist = tlsh.diff(pic1.hash, pic2.hash)
        elif HASH_CHOICE == HASH_TYPE.NO_LENGTH:
            dist = tlsh.diffxlen(pic1.hash, pic2.hash)

        return dist

# ==== Hashing ====
def hash_pictures(picture_list : List[Local_Picture]):

    for i, curr_picture in enumerate(picture_list):
        try :
            # Load and Hash picture
            hash_picture(curr_picture)
        except Exception as e :
            print("Error during hashing : " + str(e))

        if i % 40 == 0 :
            print(f"Picture {i} out of {len(picture_list)}")

    return picture_list

def hash_picture(curr_picture: Local_Picture):
    # target_hash = tlsh.hash(Image.open(curr_picture.path))
    target_hash = tlsh.hash(open(curr_picture.path, 'rb').read()) # From https://github.com/trendmicro/tlsh

    curr_picture.hash = target_hash

    if target_hash is None or target_hash == "":
        # TODO : Better handling of null hashes ?
        curr_picture.hash = '0000000000000000000000000000000000000000000000000000000000000000000000'
        raise Exception(f"Target hash null for {curr_picture.path.name}. Hash set to 0s value")

    return curr_picture

# ==== Action definition ====
class TLSH_execution_handler(execution_handler.Execution_handler) :
    def TO_OVERWRITE_prepare_dataset(self, picture_list):
        print("Hash pictures ... ")
        picture_list = hash_pictures(picture_list)
        return picture_list

    def TO_OVERWRITE_prepare_target_picture(self, target_picture):
        target_picture = hash_picture(target_picture)
        return target_picture

if __name__ == '__main__':
    # target_dir = "../../datasets/raw_phishing/"
    # filesystem_lib.clean_folder(target_dir)

    # eh = TLSH_execution_handler(target_dir=target_dir, Local_Picture=Local_Picture)
    eh = TLSH_execution_handler(Local_Picture=Local_Picture)
    # eh.do_random_test()
    eh.do_full_test()
