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
class Void_baseline(execution_handler.Execution_handler) :
    def __init__(self, conf: configuration.Default_configuration):
        super().__init__(conf)
        self.Local_Picture_class_ref = picture_class.Picture

    def TO_OVERWRITE_prepare_dataset(self, picture_list):
        self.logger.info("Hash pictures ... ")
        picture_list = self.do_nothings(picture_list)
        return picture_list

    def TO_OVERWRITE_prepare_target_picture(self, target_picture):
        target_picture = self.do_nothing(target_picture)
        return target_picture

    # ==== Hashing ====
    def do_nothings(self, picture_list : List[picture_class.Picture]):

        for i, curr_picture in enumerate(picture_list):
            self.do_nothing(curr_picture)

        return picture_list

    def do_nothing(self, curr_picture: picture_class.Picture):
        curr_picture.hash = ''

        return curr_picture

    def TO_OVERWRITE_compute_distance(self, pic1: picture_class.Picture, pic2: picture_class.Picture):
        dist = 0

        return dist