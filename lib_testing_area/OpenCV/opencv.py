# STD LIBRARIES
import os
import sys
from enum import Enum, auto
from typing import List

import cv2
import matplotlib.pyplot as plt

# PERSONAL LIBRARIES
sys.path.append(os.path.abspath(os.path.pardir))
from utility_lib import filesystem_lib
from utility_lib import picture_class
from utility_lib import execution_handler

# ENUMERATION
class DISTANCE_TYPE(Enum):
    LEN_MIN = auto()
    LEN_MAX = auto()
    RATIO_LEN = auto()

class MATCH_TYPE(Enum):
    STD = auto()
    KNN = auto()

# CONFIGURATION
DISTANCE_CHOSEN = DISTANCE_TYPE.RATIO_LEN
MATCH_CHOSEN = MATCH_TYPE.STD
MATCH_K_FOR_KNN = 2

# Crosscheck can't be activated with some option. e.g. KNN match can't work with
CROSSCHECK_DEFAULT = True
DISTANCE_INCOMPATIBLE_OPTIONS = [2]
MATCH_INCOMPATIBLE_OPTIONS = [1]
CROSSCHECK_WORKING = False if (MATCH_CHOSEN in MATCH_INCOMPATIBLE_OPTIONS or DISTANCE_CHOSEN in DISTANCE_INCOMPATIBLE_OPTIONS) else CROSSCHECK_DEFAULT

class Local_Picture(picture_class.Picture):

    def compute_distance_ext(self, pic1, pic2):
        # print("Compute description of : ")
        # print(pic1.description)
        # print(pic2.description)

        if pic1.description is None or pic2.description is None:
            return None

        # print(len(pic1.description))
        # print(len(pic2.description))
        # bfmatcher is stored in Picture local storage
        if MATCH_CHOSEN == MATCH_TYPE.STD:
            matches = self.storage.match(pic1.description, pic2.description)
        elif MATCH_CHOSEN == MATCH_TYPE.KNN :
            matches = self.storage.knnMatch(pic1.description, pic2.description, k=MATCH_K_FOR_KNN)
        else :
            raise Exception('OPENCV WRAPPER : MATCH_CHOSEN NOT CORRECT')

        matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance.  Best come first.

        print_matches(pic1, pic2, matches)

        # THREESHOLD ? TODO
        # TODO : Previously MIN, test with MEAN ?
        # TODO : Test with Mean of matches.distance .. verify what are matches distance ..

        if DISTANCE_CHOSEN == DISTANCE_TYPE.LEN_MIN:  # MIN
            dist = 1 - len(matches) / (min(len(pic1.description), len(pic2.description)))
        elif DISTANCE_CHOSEN == DISTANCE_TYPE.LEN_MAX:  # MAX
            dist = 1 - len(matches) / (max(len(pic1.description), len(pic2.description)))
        elif DISTANCE_CHOSEN == DISTANCE_TYPE.RATIO_LEN:  # RATIO + LEN
            good = self.ratio_test(matches)
            dist = 1 - len(good) / (max(len(pic1.description), len(pic2.description)))
        else :
            raise Exception('OPENCV WRAPPER : DISTANCE_CHOSEN NOT CORRECT')

        # print(dist)

        return dist

    @staticmethod
    def ratio_test(matches):
        # Apply ratio test
        good = []
        ratio = 0.75  # 0.8 in original paper.
        # print(matches)

        ''' ## Problem with can't iterate
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append([m])
        '''

        for i, m in enumerate(matches):
            if i < len(matches) - 1 and m.distance < ratio * matches[i + 1].distance:
                good.append(m)
        return good
        # print("Good :")
        # print(good)


def draw_matches(pic1 : Local_Picture, pic2 : Local_Picture, matches):
    return cv2.drawMatches(pic1.image, pic1.key_points, pic2.image, pic2.key_points, matches , None)  # Draw circles.

def save_matches(pic1 : Local_Picture, pic2 : Local_Picture, matches):
    outImg = draw_matches(pic1, pic2, matches)

def print_matches(pic1 : Local_Picture, pic2 : Local_Picture, matches):
    outImg = draw_matches(pic1, pic2, matches)
    plt.figure(figsize=(16, 16))
    plt.title('ORB Matching Points')
    plt.imshow(outImg)
    plt.show()
    input()

def print_points(img_building, key_points):
    img_building_keypoints = cv2.drawKeypoints(img_building,
                                               key_points,
                                               img_building,
                                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # Draw circles.
    plt.figure(figsize=(16, 16))
    plt.title('ORB Interest Points')
    plt.imshow(img_building_keypoints)
    plt.show()
    input()


class Matcher():
    def __init__(self):
        self.algo = cv2.ORB_create()  # SIFT, BRISK, SURF, ..
        self.bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=CROSSCHECK_WORKING)  # NORML1 (SIFT/SURF) NORML2 (SIFT/SURG) HAMMING (ORB,BRISK,
        # BRIEF) HAMMING2 (ORB WTAK=3,4)

    def train_on_images(self, picture_list: List[Local_Picture]):
        # list_description = (o.description for o in picture_list)

        # for curr_descr in list_description:
        for curr_image in picture_list:
            self.bfmatcher.add(curr_image.description)
            curr_image.storage = self.bfmatcher

        # clusters = np.array(list_description)
        # print(type(clusters))

        # Add all descriptors in the matcher
        # bfmatcher.add(clusters)

        # Train: Does nothing for BruteForceMatcher though. Otherwise, construct a "magic good datastructure" as KDTree, for example.
        self.bfmatcher.train()

    # ==== Descriptors ====
    def describe_pictures(self, picture_list: List[Local_Picture]):
        for i, curr_picture in enumerate(picture_list):
            # Load and Hash picture
            self.describe_picture(curr_picture)
            if i % 40 == 0:
                print(f"Picture {i} out of {len(picture_list)}")

        return picture_list

    def describe_picture(self, curr_picture: Local_Picture):
        try:
            # print(curr_picture.path)
            img_building = cv2.imread(str(curr_picture.path))
            img_building = cv2.cvtColor(img_building, cv2.COLOR_BGR2RGB)  # Convert from cv's BRG default color order to RGB

            key_points, description = self.algo.detectAndCompute(img_building, None)

            # Store representation information in the picture itself
            curr_picture.image = img_building
            curr_picture.key_points = key_points
            curr_picture.description = description

        except Exception as e:
            print("Error during descriptor building : " + str(e))

        return curr_picture

# ==== Action definition ====
class OpenCV_execution_handler(execution_handler.Execution_handler) :
    def TO_OVERWRITE_prepare_dataset(self):
        print(f"Describe pictures from repository {self.target_dir}... ")
        picture_list = self.storage.describe_pictures(self.picture_list)
        print("Add cluster of trained images to BFMatcher and train it ...")
        self.storage.train_on_images(picture_list)

    def TO_OVERWRITE_prepare_target_picture(self):
        self.target_picture = self.storage.describe_picture(self.target_picture)

if __name__ == '__main__':
    target_dir = "../../datasets/raw_phishing/"
    filesystem_lib.clean_folder(target_dir)

    eh = OpenCV_execution_handler(target_dir=target_dir, Local_Picture=Local_Picture)
    eh.storage = Matcher()
    # eh.do_random_test()
    eh.do_full_test()



# ==== ARCHIVES ====
'''
    matches = bfmatcher.match(target_picture.description)
    matches = sorted(matches, key=lambda x: x.distance)

    for i in range(len(matches)):
        print(matches[i].imgIdx)
'''

'''
    min = None
    min_object = None

    for curr_picture in picture_list:
        if not are_same_picture(target_picture, curr_picture) and (min is None or min > compute_distance_ext(curr_picture, target_picture)):
            min = compute_distance_ext(curr_picture, target_picture)  # TODO : Hamming .. ?
            min_object = curr_picture

    print("original picture : \t" + str(target_picture.path))
    print("min found : \t" + str(min_object.path) + " with " + str(min))
'''

''' Ratio test in Java Example
    // ratio
    test
    LinkedList < DMatch > good_matches = new
    LinkedList < DMatch > ();
    for (Iterator < MatOfDMatch > iterator = matches.iterator(); iterator.hasNext();) {
    MatOfDMatch matOfDMatch = (MatOfDMatch) iterator.next();
    if (matOfDMatch.toArray()[0].distance / matOfDMatch.toArray()[1].distance < 0.9) {
    good_matches.add(matOfDMatch.toArray()[0]);
    }
    }
'''
