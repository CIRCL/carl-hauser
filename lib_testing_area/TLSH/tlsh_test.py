# STD LIBRARIES
import os

import sys
import tlsh
from typing import List

# PERSONAL LIBRARIES
sys.path.append(os.path.abspath(os.path.pardir))
from utility_lib import filesystem_lib, picture_class, execution_handler
import configuration

# ==== Action definition ====
class TLSH_execution_handler(execution_handler.Execution_handler) :
    def __init__(self, conf: configuration.Default_configuration):
        super().__init__(conf)
        self.Local_Picture_class_ref = picture_class.Picture

    def TO_OVERWRITE_prepare_dataset(self, picture_list):
        print("Hash pictures ... ")
        picture_list = self.hash_pictures(picture_list)
        return picture_list

    def TO_OVERWRITE_prepare_target_picture(self, target_picture):
        target_picture = self.hash_picture(target_picture)
        return target_picture

    # ==== Hashing ====
    @staticmethod
    def hash_pictures(picture_list : List[picture_class.Picture]):

        for i, curr_picture in enumerate(picture_list):
            try :
                # Load and Hash picture
                TLSH_execution_handler.hash_picture(curr_picture)
            except Exception as e :
                print("Error during hashing : " + str(e))

            if i % 40 == 0 :
                print(f"Picture {i} out of {len(picture_list)}")

        return picture_list

    @staticmethod
    def hash_picture(curr_picture: picture_class.Picture):
        # target_hash = tlsh.hash(Image.open(curr_picture.path))
        target_hash = tlsh.hash(open(curr_picture.path, 'rb').read()) # From https://github.com/trendmicro/tlsh

        curr_picture.hash = target_hash

        if target_hash is None or target_hash == "":
            # TODO : Better handling of null hashes ?
            curr_picture.hash = '0000000000000000000000000000000000000000000000000000000000000000000000'
            raise Exception(f"Target hash is None or null for {curr_picture.path.name}. Hash set to 0s value")

        return curr_picture

    def TO_OVERWRITE_compute_distance(self, pic1: picture_class.Picture, pic2: picture_class.Picture):
        dist = None
        if self.conf.ALGO == configuration.ALGO_TYPE.TLSH:
            dist = tlsh.diff(pic1.hash, pic2.hash)
        elif self.conf.ALGO == configuration.ALGO_TYPE.TLSH_NO_LENGTH:
            dist = tlsh.diffxlen(pic1.hash, pic2.hash)
        else :
            raise Exception("Invalid algorithm type for TLSH execution handler during distance computing : " + str(self.conf.ALGO.name))

        return dist