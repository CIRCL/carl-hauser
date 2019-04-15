# STD LIBRARIES
import os
import sys
from typing import List

import imagehash
from PIL import Image

# PERSONAL LIBRARIES
sys.path.append(os.path.abspath(os.path.pardir))
from utility_lib import picture_class, execution_handler
import configuration


class Image_hash_execution_handler(execution_handler.Execution_handler):
    def __init__(self, conf: configuration.Default_configuration):
        super().__init__(conf)
        self.Local_Picture_class_ref = picture_class.Picture

    # ==== Action definition ====
    def TO_OVERWRITE_prepare_dataset(self, picture_list):
        self.logger.info("Hash pictures ... ")
        picture_list = self.hash_pictures(picture_list)
        return picture_list

    def TO_OVERWRITE_prepare_target_picture(self, target_picture):
        target_picture = self.hash_picture(target_picture)
        return target_picture

    # ==== Hashing ====
    def hash_pictures(self, picture_list: List[picture_class.Picture]):

        for i, curr_picture in enumerate(picture_list):
            # Load and Hash picture
            self.hash_picture(curr_picture)
            if i % 40 == 0:
                self.logger.debug(f"Picture {i} out of {len(picture_list)}")

        return picture_list

    def hash_picture(self, curr_picture: picture_class.Picture):
        try:
            if self.conf.ALGO == configuration.ALGO_TYPE.A_HASH:  # Average
                target_hash = imagehash.average_hash(Image.open(curr_picture.path))
            elif self.conf.ALGO == configuration.ALGO_TYPE.P_HASH:  # Perception
                target_hash = imagehash.phash(Image.open(curr_picture.path))
            elif self.conf.ALGO == configuration.ALGO_TYPE.P_HASH_SIMPLE:  # Perception - simple
                target_hash = imagehash.phash_simple(Image.open(curr_picture.path))
            elif self.conf.ALGO == configuration.ALGO_TYPE.D_HASH:  # D
                target_hash = imagehash.dhash(Image.open(curr_picture.path))
            elif self.conf.ALGO == configuration.ALGO_TYPE.D_HASH_VERTICAL:  # D-vertical
                target_hash = imagehash.dhash_vertical(Image.open(curr_picture.path))
            elif self.conf.ALGO == configuration.ALGO_TYPE.W_HASH:  # Wavelet
                target_hash = imagehash.whash(Image.open(curr_picture.path))
            else:
                raise Exception('IMAGEHASH WRAPPER : HASH_CHOICE NOT CORRECT')

            # TO NORMALIZE : https://fullstackml.com/wavelet-image-hash-in-python-3504fdd282b5
            curr_picture.hash = target_hash
        except Exception as e:
            self.logger.error("Error during hashing : " + str(e))

        return curr_picture

    def TO_OVERWRITE_compute_distance(self, pic1: picture_class.Picture, pic2: picture_class.Picture):
        #TODO : To review if we divide by 2 or not. * 0.5
        distance = abs(pic1.hash - pic2.hash) / (pic1.hash.hash.size )
        return distance