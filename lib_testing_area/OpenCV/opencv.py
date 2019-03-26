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

# ENUMERATION
class DISTANCE_TYPE(Enum):
    LEN_MIN = auto()
    LEN_MAX = auto()
    RATIO_LEN = auto()
    RATIO_TEST = auto() # ONLY with KNN

class MATCH_TYPE(Enum):
    STD = auto()
    KNN = auto()

class DATASTRUCT_TYPE(Enum):
    BF = auto()
    FLANN = auto()

# CONFIGURATION
DISTANCE_CHOSEN = DISTANCE_TYPE.LEN_MAX
MATCH_CHOSEN = MATCH_TYPE.STD
MATCH_K_FOR_KNN = 2
DATASTRUCT_CHOSEN = DATASTRUCT_TYPE.BF

FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)

# See options there : https://docs.opencv.org/trunk/dc/d8c/namespacecvflann.html
FLANN_INDEX_LSH = 6
# index_params= dict(algorithm = FLANN_INDEX_LSH,table_number = 6, key_size = 12,multi_probe_level = 1)
search_params = dict(checks=50)   # or pass empty dictionary

index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6)

# Crosscheck can't be activated with some option. e.g. KNN match can't work with
CROSSCHECK_DEFAULT = True
DISTANCE_INCOMPATIBLE_OPTIONS = [DISTANCE_TYPE.RATIO_LEN]
MATCH_INCOMPATIBLE_OPTIONS = [MATCH_TYPE.KNN]
CROSSCHECK_WORKING = False if (MATCH_CHOSEN in MATCH_INCOMPATIBLE_OPTIONS or DISTANCE_CHOSEN in DISTANCE_INCOMPATIBLE_OPTIONS) else CROSSCHECK_DEFAULT

print("CONFIGURATION : " + DISTANCE_CHOSEN.name + " "  + MATCH_CHOSEN.name + " " + str(MATCH_K_FOR_KNN) +" "+ str(CROSSCHECK_WORKING) +" "+DATASTRUCT_CHOSEN.name )

class Local_Picture(picture_class.Picture):

    def compute_distance_ext(self, pic1, pic2): # self, target

        if pic1.description is None or pic2.description is None:
            if pic1.description is None and pic2.description is None:
                return 0 # Pictures that have no description matches together
            else:
                return None

        # bfmatcher is stored in Picture local storage
        if MATCH_CHOSEN == MATCH_TYPE.STD:
            matches = self.storage.match(pic1.description, pic2.description)
        elif MATCH_CHOSEN == MATCH_TYPE.KNN :
            matches = self.storage.knnMatch(pic1.description, pic2.description, k=MATCH_K_FOR_KNN)
        else :
            raise Exception('OPENCV WRAPPER : MATCH_CHOSEN NOT CORRECT')

        self.matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance.  Best come first.

        # print_matches(pic1, pic2, matches)

        # THREESHOLD ? TODO
        # TODO : Previously MIN, test with MEAN ?
        # TODO : Test with Mean of matches.distance .. verify what are matches distance ..

        if DISTANCE_CHOSEN == DISTANCE_TYPE.LEN_MIN:  # MIN
            dist = 1 - len(matches) / (min(len(pic1.description), len(pic2.description)))
        elif DISTANCE_CHOSEN == DISTANCE_TYPE.LEN_MAX:  # MAX
            dist = 1 - len(matches) / (max(len(pic1.description), len(pic2.description)))
        elif DISTANCE_CHOSEN == DISTANCE_TYPE.RATIO_LEN:  # RATIO + LEN
            good = self.ratio_test_0(matches)
            dist = 1 - len(good) / (max(len(pic1.description), len(pic2.description)))
        elif DISTANCE_CHOSEN == DISTANCE_TYPE.RATIO_TEST:
            good = self.ratio_test_1(matches)
            dist = 1 - len(good) / (max(len(pic1.description), len(pic2.description)))
        else :
            raise Exception('OPENCV WRAPPER : DISTANCE_CHOSEN NOT CORRECT')

        return dist

    @staticmethod
    def ratio_test_0(matches):
        # Apply ratio test
        good = []
        ratio = 0.75  # 0.8 in original paper.
        # print(matches)

        for i, m in enumerate(matches):
            if i < len(matches) - 1 and m.distance < ratio * matches[i + 1].distance:
                good.append(m)
        return good
        # print("Good :")
        # print(good)

    @staticmethod
    def ratio_test_1(matches):
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        return good

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
            self.bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=CROSSCHECK_WORKING)
            # NORML1 (SIFT/SURF) NORML2 (SIFT/SURG) HAMMING (ORB,BRISK, # BRIEF) HAMMING2 (ORB WTAK=3,4)
        elif DATASTRUCT_CHOSEN == DATASTRUCT_TYPE.FLANN :
            # self.bfmatcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.bfmatcher = cv2.FlannBasedMatcher(index_params, search_params) # {} instead for second parameter ?

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

            # removal of picture that don't have descriptors
            if curr_picture.description is None :
                print(f"Picture {i} removed, due to lack of descriptors : {curr_picture.path.name}")
                del picture_list[i]
                # TODO : Parametered path
                os.system("cp ../../datasets/raw_phishing/"+curr_picture.path.name + " ./RESULTS_BLANKS/"+curr_picture.path.name )

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
    def TO_OVERWRITE_prepare_dataset(self, picture_list):
        print(f"Describe pictures from repository {self.target_dir}... ")
        picture_list = self.storage.describe_pictures(picture_list)

        if DATASTRUCT_CHOSEN != DATASTRUCT_TYPE.FLANN :
            # TODO : I don't get why training is not working on FLANN, while it seems doing nothing on BF ? Not a KDTree, that's why ?
            print("Add cluster of trained images to BFMatcher and train it ...")
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
            curr_width = target_picture.image.shape[1] + sorted_picture_list[i].image.shape[1]
            # We keep the largest picture
            if curr_width > max_width:
                max_width = curr_width
            # We keep the heighest picture
            total_height += max(target_picture.image.shape[0], sorted_picture_list[i].image.shape[0])

        new_im = Image.new('RGB', (max_width, total_height))
        draw = ImageDraw.Draw(new_im)

        y_offset = 0
        for i in range(0, min(NB_BEST_PICTURES,len(sorted_picture_list))):
            # Get the matches
            outImg = draw_matches(sorted_picture_list[i], target_picture, sorted_picture_list[i].matches)
            tmp_img = Image.fromarray(outImg)

            # Copy paste the matches in the column
            new_im.paste(tmp_img, (0, y_offset))

            # Print nice text
            P1 = "LEFT = BEST MATCH #" + str(i+offset) + " d=" + str(sorted_picture_list[i+offset].distance)
            P2 = " at " + sorted_picture_list[i+offset].path.name
            P3 = "| RIGHT = ORIGINAL IMAGE"
            P4 = " at " + target_picture.path.name
            tmp_title = P1 + P2 + P3 + P4
            self.text_and_outline(draw, 10, y_offset + 10, tmp_title, font_size= max_width // 60)

            y_offset += tmp_img.size[1]

        print("Save to : " + file_name)
        new_im.save(file_name)



if __name__ == '__main__':
    target_dir = "../../datasets/raw_phishing/"
    filesystem_lib.clean_folder(target_dir)

    eh = OpenCV_execution_handler(target_dir=target_dir, Local_Picture=Local_Picture, save_picture=True)
    eh.storage = Matcher()
    eh.printer = Custom_printer()
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
