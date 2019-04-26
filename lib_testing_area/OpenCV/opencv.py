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
import numpy as np

# PERSONAL LIBRARIES
sys.path.append(os.path.abspath(os.path.pardir))
from utility_lib import filesystem_lib, printing_lib, picture_class, execution_handler, json_class
import configuration
from .custom_printer import Custom_printer, Local_Picture

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
        # elif self.conf.FILTER == configuration.FILTER_TYPE.RATIO_BAD:
        #    good = self.ratio_bad(matches)
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

        if self.conf.POST_FILTER_CHOSEN == configuration.POST_FILTER.NONE :
            #Do nothing
            self.logger.debug("No post filtering")
        elif self.conf.POST_FILTER_CHOSEN == configuration.POST_FILTER.MATRIX_CHECK :
            dist = self.matrix_filtering(dist, pic1, pic2)
        else :
            raise Exception('OPENCV WRAPPER : POST_FILTER CHOSEN NOT CORRECT.')

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

    def ransac_filter(self, matches, pic1, pic2):
        '''
        Find a geomatrical transformation with RANSAC algorithms and filter outliers points thanks to the found transformation.
        Does store the transformation matrix in the source picture (pic1).
        '''

        MIN_MATCH_COUNT = 10
        good = []
        transformation_matrix = None

        # ======================= --------------------------- =======================
        #                        Filter matches to accelerate
        # Do remove the farthest matches to greatly accelerate RANSAC
        # From : http://answers.opencv.org/question/984/performance-of-findhomography/

        diminished_matches = []
        for m in matches :
            if m.distance < self.conf.RANSAC_ACCELERATOR_THRESHOLD :
                diminished_matches.append(m)

        # ======================= --------------------------- =======================
        #                        Compute homography with RANSAC

        if len(diminished_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([pic1.key_points[m.queryIdx].pt for m in diminished_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([pic2.key_points[m.trainIdx].pt for m in diminished_matches]).reshape(-1, 1, 2)

            # Find the transformation between points
            transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Get a mask list for matches = A list that says "This match is an in/out-lier"
            matchesMask = mask.ravel().tolist()

            # Filter the matches list thanks to the mask
            for i, element in enumerate(matchesMask):
                if element == 1:
                    good.append(diminished_matches[i])
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

    def matrix_filtering(self, dist, pic1, pic2):
        # Ideas from :
        # - https://stackoverflow.com/questions/10972438/detecting-garbage-homographies-from-findhomography-in-opencv/10981249#10981249
        # - https://stackoverflow.com/questions/14954220/how-to-check-if-obtained-homography-matrix-is-good?noredirect=1&lq=1
        # - https://stackoverflow.com/questions/16439792/how-can-i-compute-svd-and-and-verify-that-the-ratio-of-the-first-to-last-singula?noredirect=1&lq=1
        # - https://answers.opencv.org/question/2588/check-if-homography-is-good/

        if pic1.transformation_matrix is None :
            self.logger.error(f"NO TRANSFORMATION MATRIX FOR : {pic1.path}")
            return dist


        '''
        Compute the determinant of the homography, and see if it's too close to zero for comfort.
        Compute the determinant of the top left 2x2 homography matrix, and check if it's "too close" to zero for comfort...
        btw you can also check if it's *too *far from zero because then the invert matrix would have a determinant too close to zero.
        '''
        det = pic1.transformation_matrix[0][0] * pic1.transformation_matrix[0][0] - pic1.transformation_matrix[1][0]*pic1.transformation_matrix[0][1]

        '''
        A determinant of zero would mean the matrix is not inversible, too close to zero would mean *singular
        (like you see the plane object at 90°, which is almost impossible if you use *good matches).
        '''
        threshold = 1
        if math.fabs(det) > threshold : # or math.fabs(det) < (1.0 / threshold) :
            self.logger.warning(f"Almost 90° rotation for : {pic1.path}")
            return 1 # bad

        '''
        And while we are at it...if det<0 the homography is not conserving the orientation (clockwise<->anticlockwise), 
        except if you are watching the object in a mirror...it is certainly not good (plus the sift/surf descriptors are not done to be mirror invariants as far as i know, so you would probably don'thave good maches).
        Else if the determinant is < 0, it is orientation-reversing.
        '''
        # H.at < double > (0, 0) * H.at < double > (1, 1) - H.at < double > (1, 0) * H.at < double > (0, 1);
        if det < 0 :
            self.logger.warning(f"Mirror scene for : {pic1.path}")
            return 1 # no mirrors in the scene

        '''
        Even better, compute its SVD, and verify that the ratio of the first-to-last singular value is sane (not too high). 
        Either result will tell you whether the matrix is close to singular.
        you'd have to verify that the largest eigen-value isn't too small too.
        '''

        '''
        Compute the images of the image corners and of its center (i.e. the points you get when you apply the homography to those corners and center), 
        and verify that they make sense, i.e. are they inside the image canvas (if you expect them to be)? Are they well separated from each other?
        '''

        self.logger.info(f"Previously calculated distance : {dist}")


        # Get the size of the current matching picture
        h, w, d = pic1.image.shape
        # Get the position of the 4 corners of the current matching picture
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        max = 4 * cv2.norm(np.float32([[w,h]]), cv2.NORM_L2)
        # max = 1 if max == 0 else max

        try:
            # Transform the 4 corners thanks to the transformation matrix calculated
            dst = cv2.perspectiveTransform(pts, pic1.transformation_matrix)

            # Draw the transformed 4 corners on the target picture (pic2, request)
            tmp_dist = round(cv2.norm(pts - dst, cv2.NORM_L2) / max ,10)# sqrt((X1-X2)²+(Y1-Y2)²+...)

            self.logger.info(f"Ransac corners calculated distance : {tmp_dist}")

            dist = tmp_dist

        except Exception as e:
            self.logger.error(f"Inverting RANSAC transformation matrix impossible due to : {e} on picture {pic1.path}")


        #TODO : fill this equation
        # min(X_max - X_min) > 0.2 * width_picture

        '''
        Plot in matlab/octave the output (data) points you fitted the homography to, along with their computed values from the input ones, 
        using the homography, and verify that they are close (i.e. the error is low).
        '''

        '''
        Homography should preserve the direction of polygonal points. 
        Design a simple test. points (0,0), (imwidth,0), (width,height), (0,height) represent a quadrilateral with clockwise arranged points. 
        Apply homography on those points and see if they are still clockwise arranged if they become counter clockwise your homography is flipping (mirroring) 
        the image which is sometimes still ok. But if your points are out of order than you have a "bad homography"
        '''

        '''
        The homography doesn't change the scale of the object too much. For example if you expect it to shrink or enlarge the image by a factor of up to X, just check this rule. Transform the 4 points (0,0), (imwidth,0), (width-1,height), (0,height) with homography and calculate the area of the quadrilateral (opencv method of calculating area of polygon) if the ratio of areas is too big (or too small), you probably have an error.
        '''

        '''
        Good homography is usually uses low values of perspectivity. 
        Typically if the size of the image is ~1000x1000 pixels those values should be ~0.005-0.001. 
        High perspectivity will cause enormous distortions which are probably an error. If you don't know where those values are located read my post: 
        trying to understand the Affine Transform . It explains the affine transform math and the other 2 values are perspective parameters.
        '''

        '''
        Compute the area before/after
        '''

        return dist
