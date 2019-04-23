# STD LIBRARIES
import os
import sys
from enum import Enum, auto
from typing import List
import logging
import pathlib
import math

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
class OpenCV_execution_handler(execution_handler.Execution_handler):
    def __init__(self, conf: configuration.ORB_default_configuration):
        super().__init__(conf)
        self.Local_Picture_class_ref = Local_Picture
        self.conf = conf

        self.printer = Custom_printer(self.conf)

        # ===================================== CROSSCHECK =====================================
        # Crosscheck can't be activated with some option. e.g. KNN match can't work with
        if conf.CROSSCHECK == configuration.CROSSCHECK.AUTO:
            FILTER_INCOMPATIBLE_OPTIONS = [configuration.FILTER_TYPE.RATIO_CORRECT]
            MATCH_INCOMPATIBLE_OPTIONS = [configuration.MATCH_TYPE.KNN]
            self.CROSSCHECK = False if (self.conf.MATCH in MATCH_INCOMPATIBLE_OPTIONS or self.conf.FILTER in FILTER_INCOMPATIBLE_OPTIONS) else True
        elif conf.CROSSCHECK == configuration.CROSSCHECK.ENABLED:
            self.CROSSCHECK = True
        elif conf.CROSSCHECK == configuration.CROSSCHECK.DISABLED:
            self.CROSSCHECK = False
        else:
            raise Exception("CROSSCHECK value in configuration is wrong. Please review the value.")

        self.logger.info(f"Crosscheck selected : {self.CROSSCHECK}")

        # ===================================== ALGORITHM TYPE =====================================
        self.algo = cv2.ORB_create(nfeatures=conf.ORB_KEYPOINTS_NB)
        # SIFT, BRISK, SURF, .. # Available to change nFeatures=1000 for example. Limited to 500 by default

        # ===================================== DATASTRUCTURE =====================================
        if self.conf.DATASTRUCT == configuration.DATASTRUCT_TYPE.BRUTE_FORCE:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=self.CROSSCHECK)
            # NORML1 (SIFT/SURF) NORML2 (SIFT/SURG) HAMMING (ORB,BRISK, # BRIEF) HAMMING2 (ORB WTAK=3,4)
        elif self.conf.DATASTRUCT == configuration.DATASTRUCT_TYPE.FLANN_KDTREE:
            self.matcher = cv2.FlannBasedMatcher(conf.FLANN_KDTREE_INDEX_params, conf.FLANN_KDTREE_SEARCH_params)
        elif self.conf.DATASTRUCT == configuration.DATASTRUCT_TYPE.FLANN_LSH:
            self.matcher = cv2.FlannBasedMatcher(conf.FLANN_LSH_INDEX_params, conf.FLANN_LSH_SEARCH_params)
        else:
            raise Exception("DATASTRUCT value in configuration is wrong. Please review the value.")

    def TO_OVERWRITE_prepare_dataset(self, picture_list):
        # ===================================== PREPARE PICTURES = GIVE DESCRIPTORS =====================================
        self.logger.info(f"Describe pictures from repository {self.conf.SOURCE_DIR} ... ")
        picture_list = self.describe_pictures(picture_list)

        # ===================================== CONSTRUCT DATASTRUCTURE =====================================
        self.logger.info("Add cluster of trained images to matcher and train it ...")
        self.train_on_images(picture_list)

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
                os.system("cp " + str((self.conf.SOURCE_DIR / curr_picture.path.name).resolve()) + str(
                        pathlib.Path(" ./RESULTS_BLANKS/") / curr_picture.path.name))
            else:
                clean_picture_list.append(curr_picture)

        self.logger.info(f"New list length (without None-descriptors pictures : {len(picture_list)}")

        return clean_picture_list

    def train_on_images(self, picture_list: List[Local_Picture]):
        # TODO : ONLY KDTREE FLANN ! OR BF (but does nothing)

        # ===================================== ALL OTHER TRAINING =====================================
        # Construct a "magic good datastructure" as KDTree, for example.

        for curr_image in picture_list:
            self.matcher.add(curr_image.description)
            # TODO : To decomment ? curr_image.storage = self.matcher  # Store it to allow pictures to compute it

        # TODO : To decomment ? if self.conf.DATASTRUCT != configuration.DATASTRUCT_TYPE.FLANN_LSH:
        self.matcher.train()
        # Train: Does nothing for BruteForceMatcher though.
        # TODO : To decomment ? else:
        # TODO : To decomment ?     self.logger.warning("No training on the matcher : FLANN LSH selected.")

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
            self.logger.error("Error during descriptor building : " + str(e))

        return curr_picture

    def TO_OVERWRITE_compute_distance(self, pic1: Local_Picture, pic2: Local_Picture):  # self, target

        if pic1.description is None or pic2.description is None:
            if pic1.description is None and pic2.description is None:
                return 0  # Pictures that have no description matches together
            else:
                return None

        matches = []
        good = []

        # bfmatcher is stored in Picture local storage
        if self.conf.MATCH == configuration.MATCH_TYPE.STD:
            matches = self.matcher.match(pic1.description, pic2.description)
            # self.matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance.  Best come first.
        elif self.conf.MATCH == configuration.MATCH_TYPE.KNN:
            matches = self.matcher.knnMatch(pic1.description, pic2.description, k=self.conf.MATCH_K_FOR_KNN)
        else:
            raise Exception('OPENCV WRAPPER : MATCH_CHOSEN NOT CORRECT')

            # THREESHOLD ? TODO
            # TODO : Previously MIN, test with MEAN ?
            # TODO : Test with Mean of matches.distance .. verify what are matches distance ..

        if self.conf.FILTER == configuration.FILTER_TYPE.NO_FILTER:
            good = matches
        elif self.conf.FILTER == configuration.FILTER_TYPE.RATIO_BAD:
            good = self.ratio_bad(matches)
        elif self.conf.FILTER == configuration.FILTER_TYPE.RATIO_CORRECT:
            good = self.ratio_good(matches)
        elif self.conf.FILTER == configuration.FILTER_TYPE.FAR_THREESHOLD:
            good = self.ratio_good(matches)
            good = self.threeshold_distance_filter(good)
        # elif self.conf.FILTER == configuration.FILTER_TYPE.BASIC_THRESHOLD :
        #    good = self.threeshold_distance_filter(good)
        elif self.conf.FILTER == configuration.FILTER_TYPE.RANSAC:
            good, _ = self.ransac_filter(matches, pic1, pic2)
        else:
            raise Exception('OPENCV WRAPPER : FILTER_CHOSEN NOT CORRECT')

        if self.conf.DISTANCE == configuration.DISTANCE_TYPE.LEN_MIN:  # MIN
            dist = 1 - len(good) / (min(len(pic1.description), len(pic2.description)))
        elif self.conf.DISTANCE == configuration.DISTANCE_TYPE.LEN_MAX:  # MAX
            dist = 1 - len(good) / (max(len(pic1.description), len(pic2.description)))
        elif self.conf.DISTANCE == configuration.DISTANCE_TYPE.MEAN_DIST_PER_PAIR:
            if len(good) == 0:
                dist = None
            else:
                dist = self.mean_matches_dist(good)
        elif self.conf.DISTANCE == configuration.DISTANCE_TYPE.MEAN_AND_MAX:
            if len(good) == 0:
                dist = None
            else:
                dist1 = self.mean_matches_dist(good)
                dist2 = 1 - len(good) / (max(len(pic1.description), len(pic2.description)))
                dist = dist1 + dist2
        else:
            raise Exception('OPENCV WRAPPER : DISTANCE_CHOSEN NOT CORRECT')

            # if DISTANCE_CHOSEN == DISTANCE_TYPE.RATIO_LEN or DISTANCE_CHOSEN == DISTANCE_TYPE.RATIO_TEST :
        pic1.not_filtered_matches = sorted(matches, key=lambda x: x.distance)
        pic1.matches = sorted(good, key=lambda x: x.distance)  # Sort matches by distance.  Best come first.

        return dist

    @staticmethod
    def mean_matches_dist(matches):
        mean_dist = 0
        for curr_match in matches:
            mean_dist += curr_match.distance
        mean_dist /= len(matches)

        logging.debug(f"Current mean dist : {mean_dist}")
        return mean_dist

    @staticmethod
    def ratio_bad(matches):
        ''' NOT GOOD. DOES COMPARE DISTANCE BETWEEN PAIRS OF MATCHES IN ORDER. NO SENSE ! '''
        # Apply ratio test
        good = []
        ratio = 0.75  # 0.8 in original paper.

        for i, m in enumerate(matches):
            if i < len(matches) - 1 and m.distance < ratio * matches[i + 1].distance:
                good.append(m)
        return good

    @staticmethod
    def ratio_good(matches):
        # Apply ratio test
        good = []
        for curr_matches in matches:
            if len(curr_matches) == 0:
                continue
            elif len(curr_matches) == 1:
                good.append(curr_matches[0])
            elif curr_matches[0].distance < 0.75 * curr_matches[1].distance:
                good.append(curr_matches[0])
        return good

    @staticmethod
    def threeshold_distance_filter(matches):
        dist_th = 64
        good = []

        for curr_matches in matches:
            if curr_matches.distance < dist_th:
                good.append(curr_matches)

        return good

    @staticmethod
    def ransac_filter(matches, pic1, pic2):
        '''
        Find a geomatrical transformation with RANSAC algorithms and filter outliers points thanks to the found transformation.
        Does store the transformation matrix in the source picture (pic1).
        '''

        MIN_MATCH_COUNT = 10
        good = []
        transformation_matrix = None

        #TODO : Check =  à l'envers ?

        if len(matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([pic1.key_points[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([pic2.key_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Find the transformation between points
            transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Get a mask list for matches = A list that says "This match is an in/out-lier"
            matchesMask = mask.ravel().tolist()

            # Filter the matches list thanks to the mask
            for i, element in enumerate(matchesMask):
                if element == 1:
                    good.append(matches[i])
            # h, w = pic1.image.shape
            # pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # dst = cv2.perspectiveTransform(pts, M)

            # img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            pic1.transformation_matrix = transformation_matrix
            pic1.matchesMask = matchesMask

        else:
            logger = logging.getLogger()
            logger.info("not enough matches")

        return good, transformation_matrix


class Custom_printer(printing_lib.Printer):

    def save_pictures(self, sorted_picture_list: List[Local_Picture], target_picture: Local_Picture, file_name : pathlib.Path):

        if configuration.PICTURE_SAVE_MODE.TOP3 in self.conf.SAVE_PICTURE_INSTRUCTION_LIST:
            new_directory = file_name.parent / "TOP3"
            if not new_directory.exists() :new_directory.mkdir()
            self.save_pictures_top_3(sorted_picture_list, target_picture, file_name=new_directory / pathlib.Path(file_name.with_suffix("").name + "_top3").with_suffix(".png"))

        if configuration.PICTURE_SAVE_MODE.FEATURE_MATCHES_TOP3 in self.conf.SAVE_PICTURE_INSTRUCTION_LIST:
            new_directory = file_name.parent / "MATCHES"
            if not new_directory.exists() :new_directory.mkdir()
            self.save_pictures_matches_top_3(sorted_picture_list, target_picture, file_name=new_directory / pathlib.Path(file_name.with_suffix("").name + "_matches").with_suffix(".png"))

        if configuration.PICTURE_SAVE_MODE.FEATURE_MATCHES_TOP3 in self.conf.SAVE_PICTURE_INSTRUCTION_LIST:
            new_directory = file_name.parent / "MATCHES_INOUTLINERS"
            if not new_directory.exists() :new_directory.mkdir()
            self.save_pictures_matches_top_3_red_green(sorted_picture_list, target_picture, file_name=new_directory / pathlib.Path(file_name.with_suffix("").name + "_inoutliners").with_suffix(".png"))

        if configuration.PICTURE_SAVE_MODE.RANSAC_MATRIX in self.conf.SAVE_PICTURE_INSTRUCTION_LIST :
            if self.conf.FILTER == configuration.FILTER_TYPE.RANSAC :
                new_directory = file_name.parent / "RANSAC"
                if not new_directory.exists() :new_directory.mkdir()
                self.save_pictures_ransac(sorted_picture_list, target_picture, file_name=new_directory / pathlib.Path(file_name.with_suffix("").name + "_ransac").with_suffix(".png"))
            else :
                self.logger.warning("RANSAC filter not selected for computation, but launch configuration is asking to save RANSAC pictures. Aborting this instruction.")

    def save_pictures_matches_top_3(self, sorted_picture_list: List[Local_Picture], target_picture: Local_Picture, file_name):

        max_width = 0
        total_height = 0
        NB_BEST_PICTURES = 3

        # Preprocess to remove target picture from matches
        offset = json_class.remove_target_picture_from_matches(sorted_picture_list, target_picture)

        for i in range(0, min(NB_BEST_PICTURES, len(sorted_picture_list))):
            max_width = max(target_picture.image.shape[1] + sorted_picture_list[i + offset].image.shape[1],max_width)
            # We keep the heighest picture
            total_height += max(target_picture.image.shape[0], sorted_picture_list[i + offset].image.shape[0])

        new_im = Image.new('RGB', (max_width, total_height))
        draw = ImageDraw.Draw(new_im)

        y_offset = 0
        for i in range(0, min(NB_BEST_PICTURES, len(sorted_picture_list))):
            # Get the matches
            output = self.draw_matches(sorted_picture_list[i + offset], target_picture, sorted_picture_list[i + offset].matches)

            # img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
            # SAVE : tmp_img = Image.fromarray(outImg_good_matches)
            tmp_img = Image.fromarray(output)

            # Copy paste the matches in the column
            new_im.paste(tmp_img, (0, y_offset))

            # Print nice text
            P1 = "LEFT = BEST MATCH #" + str(i + offset) + " d=" + str(sorted_picture_list[i + offset].distance)
            P2 = " at " + sorted_picture_list[i + offset].path.name
            P3 = "| RIGHT = ORIGINAL IMAGE"
            P4 = " at " + target_picture.path.name + "\n"

            if sorted_picture_list[i + offset].description is not None:
                P5 = str(len(sorted_picture_list[i + offset].description)) + " descriptors for LEFT "
            else:
                P5 = "NONE DESCRIPTORS LEFT "

            if target_picture.description is not None:
                P6 = str(len(target_picture.description)) + " descriptors for RIGHT "
            else:
                P6 = "NONE DESCRIPTORS RIGHT "

            if sorted_picture_list[i + offset].matches is not None:
                P7 = str(len(sorted_picture_list[i + offset].matches)) + "# matches "
            else:
                P7 = "NONE MATCHES "

            tmp_title = P1 + P2 + P3 + P4 + P5 + P6 + P7
            self.text_and_outline(draw, 10, y_offset + 10, tmp_title, font_size=max_width // 60)

            y_offset += tmp_img.size[1]

        self.logger.debug(f"Save to : {str(file_name)}")
        print(file_name)

        new_im.save(file_name)

    def save_pictures_matches_top_3_red_green(self, sorted_picture_list: List[Local_Picture], target_picture: Local_Picture, file_name):

        max_width = 0
        total_height = 0
        NB_BEST_PICTURES = 3

        # Preprocess to remove target picture from matches
        offset = json_class.remove_target_picture_from_matches(sorted_picture_list, target_picture)

        for i in range(0, min(NB_BEST_PICTURES, len(sorted_picture_list))):
            max_width = max(target_picture.image.shape[1] + sorted_picture_list[i + offset].image.shape[1],max_width)
            # We keep the heighest picture
            total_height += max(target_picture.image.shape[0], sorted_picture_list[i + offset].image.shape[0])

        new_im = Image.new('RGB', (max_width, total_height))
        draw = ImageDraw.Draw(new_im)

        y_offset = 0
        for i in range(0, min(NB_BEST_PICTURES, len(sorted_picture_list))):
            # Get the matches
            output = cv2.drawMatches(sorted_picture_list[i + offset].image, sorted_picture_list[i + offset].key_points,
                                     target_picture.image, target_picture.key_points,
                                     sorted_picture_list[i + offset].not_filtered_matches, None, matchColor = (255,0,0))  # Draw circles.
            output = cv2.drawMatches(sorted_picture_list[i + offset].image, sorted_picture_list[i + offset].key_points,
                                     target_picture.image, target_picture.key_points,
                                     sorted_picture_list[i + offset].matches, output,
                                     # sorted_picture_list[i + offset].not_filtered_matches, output,
                                     matchColor = (0,255,0), flags=1)  # Draw circles.
                                     # matchesMask = sorted_picture_list[i + offset].matchesMask, matchColor = (0,255,0), flags=1)  # Draw circles.

            tmp_img = Image.fromarray(output)

            # Copy paste the matches in the column
            new_im.paste(tmp_img, (0, y_offset))

            # Print nice text
            P1 = "LEFT = BEST MATCH #" + str(i + offset) + " d=" + str(sorted_picture_list[i + offset].distance)
            P2 = " at " + sorted_picture_list[i + offset].path.name
            P3 = "| RIGHT = ORIGINAL IMAGE"
            P4 = " at " + target_picture.path.name + "\n"

            if sorted_picture_list[i + offset].description is not None:
                P5 = str(len(sorted_picture_list[i + offset].description)) + " descriptors for LEFT "
            else:
                P5 = "NONE DESCRIPTORS LEFT "

            if target_picture.description is not None:
                P6 = str(len(target_picture.description)) + " descriptors for RIGHT "
            else:
                P6 = "NONE DESCRIPTORS RIGHT "

            if sorted_picture_list[i + offset].matches is not None:
                P7 = str(len(sorted_picture_list[i + offset].matches)) + "# matches "
            else:
                P7 = "NONE MATCHES "

            tmp_title = P1 + P2 + P3 + P4 + P5 + P6 + P7
            self.text_and_outline(draw, 10, y_offset + 10, tmp_title, font_size=max_width // 60)

            y_offset += tmp_img.size[1]

        self.logger.debug(f"Save to : {str(file_name)}")
        print(file_name)

        new_im.save(file_name)

    @staticmethod
    def draw_matches(pic1: Local_Picture, pic2: Local_Picture, matches):
        return cv2.drawMatches(pic1.image, pic1.key_points, pic2.image, pic2.key_points, matches, None)  # Draw circles.

    @staticmethod
    def save_matches(pic1: Local_Picture, pic2: Local_Picture, matches, distance):
        outImg = Custom_printer.draw_matches(pic1, pic2, matches)
        # outImg = printing_lib.print_title(outImg, pic1.path.name + " " + pic2.path.name)
        logging.debug("./RESULTS/" + pic1.path.name)
        t = pic1.path.name + " TO " + pic1.path.name + " IS " + str(distance)
        plt.text(0, 0, t, ha='center', wrap=True)
        plt.imsave("./RESULTS/" + pic1.path.name, outImg)

    @staticmethod
    def print_matches(pic1: Local_Picture, pic2: Local_Picture, matches):
        outImg = Custom_printer.draw_matches(pic1, pic2, matches)
        plt.figure(figsize=(16, 16))
        plt.title('ORB Matching Points')
        plt.imshow(outImg)
        plt.show()
        # input()

    @staticmethod
    def print_points(img_building, key_points):
        img_building_keypoints = cv2.drawKeypoints(img_building,
                                                   key_points,
                                                   img_building,
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # Draw circles.
        plt.figure(figsize=(16, 16))
        plt.title('ORB Interest Points')
        plt.imshow(img_building_keypoints)
        plt.show()
        # input()

    def save_pictures_ransac(self, sorted_picture_list: List[Local_Picture], target_picture: Local_Picture, file_name):

        # TODO : Print transformation

        max_width = 0
        total_height = 0
        NB_BEST_PICTURES = 3

        # Preprocess to remove target picture from matches
        offset = json_class.remove_target_picture_from_matches(sorted_picture_list, target_picture)

        for i in range(0, min(NB_BEST_PICTURES, len(sorted_picture_list))):
            # We keep the largest picture
            max_width = max(target_picture.image.shape[1]*2 + sorted_picture_list[i + offset].image.shape[1],max_width) # *2 for the skewed/deskewed
            # We keep the heighest picture
            total_height += max(target_picture.image.shape[0], sorted_picture_list[i + offset].image.shape[0])

        new_im = Image.new('RGB', (max_width, total_height))
        draw = ImageDraw.Draw(new_im)

        y_offset = 0
        for i in range(0, min(NB_BEST_PICTURES, len(sorted_picture_list))):
            trans_matrix = sorted_picture_list[i + offset].transformation_matrix

            # Get the size of the current matching picture
            h, w, d = sorted_picture_list[i + offset].image.shape
            # Get the position of the 4 corners of the current matching picture
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            try :
                # Transform the 4 corners thanks to the transformation matrix calculated
                dst = cv2.perspectiveTransform(pts, trans_matrix)
            except Exception as e :
                self.logger.error(f"Inverting RANSAC transformation matrix impossible due to : {e} on picture {sorted_picture_list[i + offset].path}. Is RANSAC the chosen filter ?")
                continue

            # Draw the transformed 4 corners on the target picture (pic2, request)
            img2 = cv2.polylines(target_picture.image, [np.int32(dst)], True, 255, 7, cv2.LINE_AA)

            # Draw matches side to side between picture reference and picture request
            output = cv2.drawMatches(sorted_picture_list[i + offset].image, sorted_picture_list[i + offset].key_points,
                                     img2, target_picture.key_points,
                                     sorted_picture_list[i + offset].matches, None)

            # Compute the transformation between picture reference and picture request (scale, and 3D angle)
            # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
            ss = trans_matrix[0, 1]
            sc = trans_matrix[0, 0]
            scaleRecovered = math.sqrt(ss * ss + sc * sc)
            thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
            self.logger.debug(f"MAP: Calculated scale difference: {scaleRecovered}, Calculated rotation difference: {thetaRecovered}" )

            # Deskew target picture (request picture) into a "transformed" version
            skewed_picture = target_picture.image
            orig_picture = sorted_picture_list[i + offset].image
            im_out = cv2.warpPerspective(skewed_picture, np.linalg.inv(trans_matrix),(orig_picture.shape[1], orig_picture.shape[0]))

            # Copy paste the picture in the right global position of the output picture
            new_im.paste(Image.fromarray(output), (0, y_offset)) # height offset
            new_im.paste(Image.fromarray(im_out), (output.shape[1], y_offset)) # width offset and height offset

            # Print nice text
            P1 = "LEFT = BEST MATCH #" + str(i + offset) + " d=" + str(sorted_picture_list[i + offset].distance)
            P2 = " at " + sorted_picture_list[i + offset].path.name
            P3 = "| RIGHT = ORIGINAL IMAGE"
            P4 = " at " + target_picture.path.name + "\n"

            if sorted_picture_list[i + offset].description is not None:
                P5 = str(len(sorted_picture_list[i + offset].description)) + " descriptors for LEFT "
            else:
                P5 = "NONE DESCRIPTORS LEFT "

            if target_picture.description is not None:
                P6 = str(len(target_picture.description)) + " descriptors for RIGHT "
            else:
                P6 = "NONE DESCRIPTORS RIGHT "

            if sorted_picture_list[i + offset].matches is not None:
                P7 = str(len(sorted_picture_list[i + offset].matches)) + "# matches "
            else:
                P7 = "NONE MATCHES "

            tmp_title = P1 + P2 + P3 + P4 + P5 + P6 + P7
            self.text_and_outline(draw, 10, y_offset + 10, tmp_title, font_size=max_width // 60)

            y_offset += Image.fromarray(output).size[1]

        self.logger.debug(f"Save to : {str(file_name)}")
        print(file_name)

        new_im.save(file_name)
