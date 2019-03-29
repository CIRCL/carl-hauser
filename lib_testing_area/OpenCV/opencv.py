# STD LIBRARIES
import os
import sys
from enum import Enum, auto
from typing import List

import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# PERSONAL LIBRARIES
sys.path.append(os.path.abspath(os.path.pardir))
from utility_lib import filesystem_lib, printing_lib, picture_class, execution_handler, json_class


# POSSIBLE CONFIGURATION :
# ALL DIST with NO FILTER with STD with BF
# ALL DIST with RATIO BAD with STD with BF

# ALL DIST with NO FILTER with KNN with FLANN_LSH : CRASH
# ALL DIST with RATIO CORRECT with KNN with FLANN_LSH : OK
# ALL DIST with FAR_THREESHOLD with KNN with FLANN_LSH : OK

# ENUMERATION
class DISTANCE_TYPE(Enum):
    LEN_MIN = auto()
    LEN_MAX = auto()
    LEN_MEAN = auto()

class FILTER_TYPE(Enum):
    RATIO_BAD = auto() # NOT with KNN
    RATIO_CORRECT = auto() # ONLY with KNN
    FAR_THREESHOLD = auto() # NOT with KNN
    NO_FILTER = auto()

class MATCH_TYPE(Enum):
    STD = auto()
    KNN = auto()

class DATASTRUCT_TYPE(Enum):
    BF = auto()
    FLANN_KDTREE = auto()
    FLANN_LSH = auto()

# CONFIGURATION
DISTANCE_CHOSEN = DISTANCE_TYPE.LEN_MAX

MATCH_CHOSEN = MATCH_TYPE.STD
MATCH_K_FOR_KNN = 2

DATASTRUCT_CHOSEN = DATASTRUCT_TYPE.BF

FILTER_CHOSEN = FILTER_TYPE.NO_FILTER

FLANN_KDTREE_INDEX = 0
FLANN_KDTREE_INDEX_params = dict(algorithm=FLANN_KDTREE_INDEX, trees=5)
FLANN_KDTREE_SEARCH_params = dict(checks=50)

# See options there : https://docs.opencv.org/trunk/dc/d8c/namespacecvflann.html
FLANN_LSH_INDEX = 6
FLANN_LSH_INDEX_params = dict(algorithm = FLANN_LSH_INDEX,table_number = 6, key_size = 12,multi_probe_level = 1)
FLANN_LSH_SEARCH_params = dict(checks=50)   # or pass empty dictionary

FLANN_LSH_INDEX_params_light = dict(algorithm = FLANN_LSH_INDEX, table_number = 6)

# Crosscheck can't be activated with some option. e.g. KNN match can't work with
CROSSCHECK_DEFAULT = True
FILTER_INCOMPATIBLE_OPTIONS = [FILTER_TYPE.RATIO_CORRECT]
MATCH_INCOMPATIBLE_OPTIONS = [MATCH_TYPE.KNN]
CROSSCHECK_WORKING = False if (MATCH_CHOSEN in MATCH_INCOMPATIBLE_OPTIONS or FILTER_CHOSEN in FILTER_INCOMPATIBLE_OPTIONS) else CROSSCHECK_DEFAULT

print("CONFIGURATION : " + DISTANCE_CHOSEN.name + " "  + MATCH_CHOSEN.name + " " + str(MATCH_K_FOR_KNN) +" "+ str(CROSSCHECK_WORKING) +" "+DATASTRUCT_CHOSEN.name +" "+FILTER_CHOSEN.name )

class Local_Picture(picture_class.Picture):

    def compute_distance_ext(self, pic1, pic2): # self, target

        if pic1.description is None or pic2.description is None:
            if pic1.description is None and pic2.description is None:
                return 0 # Pictures that have no description matches together
            else:
                return None

        matches = []
        good = []

        # bfmatcher is stored in Picture local storage
        if MATCH_CHOSEN == MATCH_TYPE.STD:
            matches = self.storage.match(pic1.description, pic2.description)
            # self.matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance.  Best come first.
        elif MATCH_CHOSEN == MATCH_TYPE.KNN :
            matches = self.storage.knnMatch(pic1.description, pic2.description, k=MATCH_K_FOR_KNN)
        else :
            raise Exception('OPENCV WRAPPER : MATCH_CHOSEN NOT CORRECT')

        # THREESHOLD ? TODO
        # TODO : Previously MIN, test with MEAN ?
        # TODO : Test with Mean of matches.distance .. verify what are matches distance ..

        if FILTER_CHOSEN == FILTER_TYPE.NO_FILTER :
            good = matches
        elif FILTER_CHOSEN == FILTER_TYPE.RATIO_BAD :
            good = self.ratio_bad(matches)
        elif FILTER_CHOSEN == FILTER_TYPE.RATIO_CORRECT :
            good = self.ratio_good(matches)
        elif FILTER_CHOSEN == FILTER_TYPE.FAR_THREESHOLD :
            good = self.ratio_good(matches)
            good = self.far_filter(good)
        else :
            raise Exception('OPENCV WRAPPER : FILTER_CHOSEN NOT CORRECT')

        if DISTANCE_CHOSEN == DISTANCE_TYPE.LEN_MIN:  # MIN
            dist = 1 - len(good) / (min(len(pic1.description), len(pic2.description)))
        elif DISTANCE_CHOSEN == DISTANCE_TYPE.LEN_MAX:  # MAX
            dist = 1 - len(good) / (max(len(pic1.description), len(pic2.description)))
        else :
            raise Exception('OPENCV WRAPPER : DISTANCE_CHOSEN NOT CORRECT')

        # if DISTANCE_CHOSEN == DISTANCE_TYPE.RATIO_LEN or DISTANCE_CHOSEN == DISTANCE_TYPE.RATIO_TEST :
        self.matches = sorted(good, key=lambda x: x.distance)  # Sort matches by distance.  Best come first.

        return dist

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
        for curr_matches in matches :
            if len(curr_matches) == 0 :
                continue
            elif len(curr_matches) == 1 :
                good.append(curr_matches[0])
            elif curr_matches[0].distance < 0.75 * curr_matches[1].distance:
                good.append(curr_matches[0])
        return good

    @staticmethod
    def far_filter(matches):
        dist_th = 64
        good = []

        for curr_matches in matches :
            if curr_matches.distance < dist_th :
                good.append(curr_matches)

        return good

    def load_image(self, path):
        if path is None or path == "" :
            raise Exception("Path specified void")
            return None

        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from cv's BRG default color order to RGB

        return image


def draw_matches(pic1 : Local_Picture, pic2 : Local_Picture, matches):
    return cv2.drawMatches(pic1.image, pic1.key_points, pic2.image, pic2.key_points, matches , None)  # Draw circles.

def save_matches(pic1 : Local_Picture, pic2 : Local_Picture, matches, distance):
    outImg = draw_matches(pic1, pic2, matches)
    # outImg = printing_lib.print_title(outImg, pic1.path.name + " " + pic2.path.name)
    print("./RESULTS/" + pic1.path.name)
    t = pic1.path.name + " TO " + pic1.path.name + " IS " + str(distance)
    plt.text(0,0, t, ha='center', wrap=True)
    plt.imsave("./RESULTS/" + pic1.path.name, outImg)

def print_matches(pic1 : Local_Picture, pic2 : Local_Picture, matches):
    outImg = draw_matches(pic1, pic2, matches)
    plt.figure(figsize=(16, 16))
    plt.title('ORB Matching Points')
    plt.imshow(outImg)
    plt.show()
    # input()

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


class Matcher():
    def __init__(self):
        self.algo = cv2.ORB_create() # nfeatures=5000)  # SIFT, BRISK, SURF, .. # Available to change nFeatures=1000 ? Limited to 500 by default

        if DATASTRUCT_CHOSEN == DATASTRUCT_TYPE.BF :
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=CROSSCHECK_WORKING)
            # NORML1 (SIFT/SURF) NORML2 (SIFT/SURG) HAMMING (ORB,BRISK, # BRIEF) HAMMING2 (ORB WTAK=3,4)
        elif DATASTRUCT_CHOSEN == DATASTRUCT_TYPE.FLANN_KDTREE :
            self.matcher = cv2.FlannBasedMatcher(FLANN_KDTREE_INDEX_params, FLANN_KDTREE_SEARCH_params)
        elif DATASTRUCT_CHOSEN == DATASTRUCT_TYPE.FLANN_LSH :
            self.matcher = cv2.FlannBasedMatcher(FLANN_LSH_INDEX_params, FLANN_LSH_SEARCH_params)

    def train_on_images(self, picture_list: List[Local_Picture]):
        #TODO : ONLY KDTREE FLANN ! OR BF (but does nothing)
        if DATASTRUCT_CHOSEN != DATASTRUCT_TYPE.FLANN_LSH :
            print("No training on the matcher : FLANN LSH chosen.")

        for curr_image in picture_list:
            self.matcher.add(curr_image.description)
            curr_image.storage = self.matcher

        if DATASTRUCT_CHOSEN != DATASTRUCT_TYPE.FLANN_LSH:
            self.matcher.train()
        # Train: Does nothing for BruteForceMatcher though. Otherwise, construct a "magic good datastructure" as KDTree, for example.

    # ==== Descriptors ====
    def describe_pictures(self, picture_list: List[Local_Picture]):
        clean_picture_list = []

        for i, curr_picture in enumerate(picture_list):

            # Load and Hash picture
            self.describe_picture(curr_picture)
            if i % 40 == 0:
                print(f"Picture {i} out of {len(picture_list)}")

            # removal of picture that don't have descriptors
            if curr_picture.description is None :
                print(f"Picture {i} removed, due to lack of descriptors : {curr_picture.path.name}")
                # del picture_list[i]

                # TODO : Parametered path
                os.system("cp ../../datasets/raw_phishing/"+curr_picture.path.name + " ./RESULTS_BLANKS/"+curr_picture.path.name )
            else :
                clean_picture_list.append(curr_picture)

        print(f"New list length (without None-descriptors pictures : {len(picture_list)}")

        return clean_picture_list

    def describe_picture(self, curr_picture: Local_Picture):
        try:
            # Picture loading handled in picture load_image overwrite
            key_points, description = self.algo.detectAndCompute(curr_picture.image, None)

            # Store representation information in the picture itself
            curr_picture.key_points = key_points
            curr_picture.description = description

            '''
            if key_points is None :
                print(f"WARNING : picture {curr_picture.path.name} has no keypoints")
            if description is None :
                print(f"WARNING : picture {curr_picture.path.name} has no description")
            '''

        except Exception as e:
            print("Error during descriptor building : " + str(e))

        return curr_picture

# ==== Action definition ====


class OpenCV_execution_handler(execution_handler.Execution_handler) :
    def TO_OVERWRITE_prepare_dataset(self, picture_list):
        print(f"Describe pictures from repository {self.target_dir} ... ")
        #TODO : WHERE IS THIS PICTURE ?? !
        picture_list = self.storage.describe_pictures(picture_list)

        print("Add cluster of trained images to matcher and train it ...")
        self.storage.train_on_images(picture_list)

        return picture_list

    def TO_OVERWRITE_prepare_target_picture(self, target_picture):
        target_picture = self.storage.describe_picture(target_picture)
        return target_picture

class Custom_printer(printing_lib.Printer):

    def save_picture_top_matches(self, sorted_picture_list: List[Local_Picture], target_picture: Local_Picture, file_name='test.png'):

        max_width = 0
        total_height = 0
        NB_BEST_PICTURES = 3

        # Preprocess to remove target picture from matches
        offset = json_class.remove_target_picture_from_matches(sorted_picture_list,target_picture)

        for i in range(0, min(NB_BEST_PICTURES,len(sorted_picture_list))):
            curr_width = target_picture.image.shape[1] + sorted_picture_list[i+offset].image.shape[1]
            # We keep the largest picture
            if curr_width > max_width:
                max_width = curr_width
            # We keep the heighest picture
            total_height += max(target_picture.image.shape[0], sorted_picture_list[i+offset].image.shape[0])

        new_im = Image.new('RGB', (max_width, total_height))
        draw = ImageDraw.Draw(new_im)

        y_offset = 0
        for i in range(0, min(NB_BEST_PICTURES,len(sorted_picture_list))):
            # Get the matches
            outImg = draw_matches(sorted_picture_list[i+offset], target_picture, sorted_picture_list[i+offset].matches)
            # img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
            tmp_img = Image.fromarray(outImg)

            # Copy paste the matches in the column
            new_im.paste(tmp_img, (0, y_offset))

            # Print nice text
            P1 = "LEFT = BEST MATCH #" + str(i+offset) + " d=" + str(sorted_picture_list[i+offset].distance)
            P2 = " at " + sorted_picture_list[i+offset].path.name
            P3 = "| RIGHT = ORIGINAL IMAGE"
            P4 = " at " + target_picture.path.name + "\n"

            if sorted_picture_list[i+offset].description is not None :
                P5 =  str(len(sorted_picture_list[i+offset].description)) + " descriptors for LEFT "
            else :
                P5 = "NONE DESCRIPTORS LEFT "

            if target_picture.description is not None:
                P6 =  str(len(target_picture.description)) + " descriptors for RIGHT "
            else:
                P6 = "NONE DESCRIPTORS RIGHT "

            if sorted_picture_list[i+offset].matches is not None:
                P7 = str(len(sorted_picture_list[i+offset].matches)) + "# matches "
            else :
                P7 = "NONE MATCHES "

            tmp_title = P1 + P2 + P3 + P4 + P5 + P6 + P7
            self.text_and_outline(draw, 10, y_offset + 10, tmp_title, font_size= max_width // 60)

            y_offset += tmp_img.size[1]

        print("Save to : " + file_name)
        new_im.save(file_name)



if __name__ == '__main__':
    target_dir = "../../datasets/raw_phishing/"
    filesystem_lib.clean_folder(target_dir)

    eh = OpenCV_execution_handler(target_dir=target_dir, Local_Picture=Local_Picture, save_picture=False)
    eh.storage = Matcher()
    eh.printer = Custom_printer()
    # eh.do_random_test()
    eh.do_full_test()