# STD LIBRARIES
import os
import sys
from enum import Enum, auto
from typing import List
import logging
import pathlib


import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

# PERSONAL LIBRARIES
sys.path.append(os.path.abspath(os.path.pardir))
from utility_lib import filesystem_lib, printing_lib, picture_class, execution_handler, json_class
import configuration

class Local_Picture(picture_class.Picture):

    def load_image(self, path):
        if path is None or path == "":
            raise Exception("Path specified void")
            return None
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from cv's BRG default color order to RGB

        return image

# ==== Action definition ====
class BoW_execution_handler(execution_handler.Execution_handler):
    def __init__(self, conf: configuration.BoW_ORB_default_configuration):
        super().__init__(conf)
        self.Local_Picture_class_ref = Local_Picture
        self.conf = conf

        # self.printer = Custom_printer(self.conf)

        # ===================================== ALGORITHM TYPE =====================================
        self.algo = cv2.ORB_create(nfeatures=conf.ORB_KEYPOINTS_NB)
        # SIFT, BRISK, SURF, .. # Available to change nFeatures=1000 for example. Limited to 500 by default

        # ===================================== DATASTRUCTURE : BoW =====================================
        self.bow_trainer = cv2.BOWKMeansTrainer(self.conf.BOW_SIZE)
        self.bow_descriptor = cv2.BOWImgDescriptorExtractor(self.algo, cv2.BFMatcher(cv2.NORM_HAMMING))

    def TO_OVERWRITE_prepare_dataset(self, picture_list):

        # ===================================== PREPARE PICTURES = GIVE DESCRIPTORS =====================================
        self.logger.info(f"Describe pictures from repository {self.conf.SOURCE_DIR} ... ")
        picture_list = self.describe_pictures(picture_list)

        # ===================================== CONSTRUCT DATASTRUCTURE =====================================
        self.logger.info("Add cluster of trained images to matcher and train it ...")
        self.train_on_images(picture_list)

        # ===================================== DESCRIBE PICTURES WITH VOCABULARY =====================================
        self.logger.info("Describe pictues with created vocabulary ...")
        picture_list = self.describe_pictures_with_vocabulary(picture_list)

        return picture_list

    # ==== Descriptors ====
    def describe_pictures(self, picture_list: List[Local_Picture]):
        clean_picture_list = []

        for i, curr_picture in enumerate(picture_list):

            # ===================================== GIVE DESCRIPTORS FOR ONE PICTURE =====================================
            self.describe_picture(curr_picture)

            if i % 40 == 0:
                self.logger.info(f"Picture {i} out of {len(picture_list)}")

            # ===================================== REMOVING EDGE CASE PICTURES =====================================
            # removal of picture that don't have descriptors
            if curr_picture.description is None:
                self.logger.warning(f"Picture {i} removed, due to lack of descriptors : {curr_picture.path.name}")
                # del picture_list[i]

                # TODO : Parametered path
                os.system("cp " + str((self.conf.SOURCE_DIR / curr_picture.path.name).resolve() ) + str(pathlib.Path(" ./RESULTS_BLANKS/") / curr_picture.path.name))
            else:
                clean_picture_list.append(curr_picture)

        self.logger.info(f"New list length (without None-descriptors pictures : {len(picture_list)}")

        return clean_picture_list

    def train_on_images(self, picture_list: List[Local_Picture]):
        # TODO : ONLY KDTREE FLANN ! OR BF (but does nothing)

        # ===================================== BOW TRAINING =====================================
        for curr_image in picture_list:
            self.bow_trainer.add(np.float32(curr_image.description))

        self.vocab = self.bow_trainer.cluster().astype(picture_list[0].description.dtype)
        self.bow_descriptor.setVocabulary(self.vocab)


    def describe_pictures_with_vocabulary(self, picture_list: List[Local_Picture]):

        # Compute new description given vocabulary
        for i, curr_picture in enumerate(picture_list):
            # keypoints = detector.detect(img, None)
            description = self.bow_descriptor.compute(curr_picture.image, curr_picture.key_points)
            curr_picture.description = description

        return picture_list

    def TO_OVERWRITE_prepare_target_picture(self, target_picture):
        target_picture = self.describe_picture(target_picture)
        return target_picture

    def describe_picture(self, curr_picture: Local_Picture):
        try:
            # Picture loading handled in picture load_image overwrite
            key_points, description = self.algo.detectAndCompute(curr_picture.image, None)

            # Store representation information in the picture itself
            curr_picture.key_points = key_points
            curr_picture.description = description

            if key_points is None:
                self.logger.warning(f"WARNING : picture {curr_picture.path.name} has no keypoints")
            if description is None:
                self.logger.warning(f"WARNING : picture {curr_picture.path.name} has no description")

        except Exception as e:
            self.logger.warning("Error during descriptor building : " + str(e))

        return curr_picture

    def TO_OVERWRITE_compute_distance(self, pic1: Local_Picture, pic2: Local_Picture):  # self, target

        if pic1.description is None or pic2.description is None:
            if pic1.description is None and pic2.description is None:
                return 0  # Pictures that have no description matches together
            else:
                return None

        # See : https://docs.opencv.org/2.4.13.7/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
        if self.conf.BOW_CMP_HIST == configuration.BOW_CMP_HIST.CORREL :
            dist = 1 - cv2.compareHist(pic1.description, pic2.description, cv2.HISTCMP_CORREL)
        elif self.conf.BOW_CMP_HIST == configuration.BOW_CMP_HIST.BHATTACHARYYA :
            dist = cv2.compareHist(pic1.description, pic2.description, cv2.HISTCMP_BHATTACHARYYA)
        else:
            raise Exception('BOW WRAPPER : HISTOGRAM COMPARISON MODE INCORRECT')

        return dist