import pathlib
import random
import time
import operator
from PIL import Image, ImageFont, ImageDraw
from scipy import stats
from typing import List
import json

import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

json_to_export = {}
hash_list = []
save_picture = False
orb = None
bfmatcher = None


class Picture():
    def __init__(self, id, shape="image", path=None):
        self.id = id
        self.shape = shape
        self.path = path
        self.matched = False
        self.sorted_matching_picture_list = []
        # Hashing related attributes
        self.key_points = None
        self.description = None
        self.distance = None

    def to_node_json_object(self):
        tmp_obj = {}
        tmp_obj["id"] = self.id
        tmp_obj["shape"] = self.shape
        tmp_obj["image"] = self.path.name
        return tmp_obj

    def compute_distance(self, target_picture):
        self.distance = compute_distance_ext(self, target_picture)
        # MIGHT BE VOID ? WARNING !

        return self.distance


def compute_distance_ext(pic1, pic2):
    # print("Compute description of : ")
    # print(pic1.description)
    # print(pic2.description)

    if pic1.description is None or pic2.description is None :
        return 1000

    # print(len(pic1.description))
    # print(len(pic2.description))

    matches = bfmatcher.match(pic1.description, pic2.description)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance.  Best come first.

    # print("Matches " + str(pic1.path) + " to " + str(pic2.path))
    # THREESHOLD ? TODO

    dist = 1 - len(matches)/(min(len(pic1.description), len(pic2.description)))
    # print(dist)

    # Apply ratio test
    '''
        good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    print("Good :")
    print(good)
    '''

    return dist


# ==== Disk access ====
def fixed_choice():
    target_picture_path = pathlib.Path('../../datasets/raw_phishing/comunidadejesusteama.org.br.png')

    return target_picture_path


def random_choice(target_dir):
    pathlist = pathlib.Path(target_dir).glob('**/*.png')
    target_picture_path = random.choice(list(pathlist))

    return target_picture_path


def get_Pictures_from_directory(directory_path):
    pathlist = pathlib.Path(directory_path).glob('**/*.png')
    picture_list = []

    for i, path in enumerate(pathlist):
        tmp_Picture = Picture(id=i, path=path)

        # Store hash
        picture_list.append(tmp_Picture)

    return picture_list


# ==== Descriptors ====
def describe_pictures(picture_list: List[Picture]):
    for curr_picture in picture_list:
        # Load and Hash picture
        curr_picture = describe_picture(curr_picture)

    return picture_list


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


def describe_picture(curr_picture: Picture):
    try:
        # print(curr_picture.path)
        img_building = cv2.imread(str(curr_picture.path))
        img_building = cv2.cvtColor(img_building, cv2.COLOR_BGR2RGB)  # Convert from cv's BRG default color order to RGB

        key_points, description = orb.detectAndCompute(img_building, None)

        # DEBUG purposes
        # print_points(img_building,key_points)

        curr_picture.key_points = key_points
        curr_picture.description = description

    except Exception as e:
        print("Error during descriptor building : " + str(e))

    return curr_picture


# ==== Checking ====
def find_closest(picture_list: List[Picture], target_picture: Picture):

    min = None
    min_object = None


    for curr_picture in picture_list:
        curr_dist = compute_distance_ext(curr_picture, target_picture)

        if not are_same_picture(target_picture, curr_picture) and (min is None or min > curr_dist):
            min = curr_dist
            min_object = curr_picture

    print("original picture : \t" + str(target_picture.path))
    print("min found : \t" + str(min_object.path) + " with " + str(min))

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





def get_top(picture_list: List[Picture], target_picture: Picture):
    for curr_picture in picture_list:
        curr_picture.compute_distance(target_picture)

    sorted_picture_list = sorted(picture_list, key=operator.attrgetter('distance'))
    print(sorted_picture_list)

    return sorted_picture_list


def print_list(hash_list):
    THREESHOLD = 5
    for i in hash_list[0:THREESHOLD]:
        print(str(i.path) + " : " + str(i.distance))


def are_same_picture(pic1: Picture, pic2: Picture):
    # TODO : Except on SHA1 hash ?
    return pic1.path == pic2.path


def remove_target_picture_from_matches(sorted_picture_list: List[Picture], target_picture: Picture):
    offset = 0
    if are_same_picture(sorted_picture_list[0], target_picture):
        # If first picture is the original picture we skip.
        print("Removed first choice : " + sorted_picture_list[0].path.name)
        offset += 1

    return offset


def json_add_nodes(picture_list: List[Picture]):
    nodes_list = []

    # Add all nodes
    for curr_picture in picture_list:
        nodes_list.append(curr_picture.to_node_json_object())

    json_to_export["nodes"] = nodes_list


TOP_K_EDGE = 1


def json_add_top_matches(sorted_picture_list: List[Picture], target_picture: Picture):
    # Preprocess to remove target picture from matches
    offset = remove_target_picture_from_matches(sorted_picture_list, target_picture)

    # Get current list of matches
    edges_list = json_to_export.get("edges", [])

    # Add all edges with labels
    for i in range(0, TOP_K_EDGE):
        tmp_obj = {}
        tmp_obj["from"] = target_picture.id
        tmp_obj["to"] = sorted_picture_list[i + offset].id
        tmp_obj["label"] = "rank " + str(i) + "(" + str(sorted_picture_list[i + offset].distance) + ")"
        edges_list.append(tmp_obj)

    # Store in JSON variable
    json_to_export["edges"] = edges_list


def json_export(json_to_export, file_name='test.json'):
    with open(pathlib.Path(file_name), 'w') as outfile:
        json.dump(json_to_export, outfile)

def train_on_images(picture_list : List[Picture]):
    list_description = (o.description for o in picture_list)

    global bfmatcher
    for curr_descr in list_description :
        bfmatcher.add(curr_descr)

    # clusters = np.array(list_description)
    # print(type(clusters))

    # Add all descriptors in the matcher
    # bfmatcher.add(clusters)

    # Train: Does nothing for BruteForceMatcher though. Otherwise, construct a "magic good datastructure" as KDTree, for example.
    bfmatcher.train()

def full_test(target_dir):
    global orb
    global bfmatcher

    orb = cv2.ORB_create()  # SIFT, BRISK, SURF, ..
    bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # NORML1 (SIFT/SURF) NORML2 (SIFT/SURG) HAMMING (ORB,BRISK,BRIEF) HAMMING2 (ORB WTAK=3,4)

    output_dir = "./RESULTS/"

    print("Load pictures ... ")
    picture_list = get_Pictures_from_directory(target_dir)

    print("Prepare Json file with initial nodes")
    json_add_nodes(picture_list)

    start_time = time.time()
    print(f"Describe pictures from repository {target_dir}... ")
    picture_list = describe_pictures(picture_list)
    elapsed = time.time() - start_time
    print(f"Elapsed hashing time : {elapsed} sec for {len(picture_list)} items ({elapsed / len(picture_list)} per item)")

    print("Add cluster of trained images to BFMatcher and train it ...")
    train_on_images(picture_list)

    start_FULL_time = time.time()
    list_time = []
    for i, curr_target_picture in enumerate(picture_list):
        start_time = time.time()
        print(f"PICTURE {i} picked as target ... ")
        print("Target picture : " + str(curr_target_picture.path))

        print("Find closest picture from target picture ... ")
        # find_closest(picture_list, curr_target_picture) # Only a print

        print("Extract top K images ... ")
        sorted_picture_list = get_top(picture_list, curr_target_picture)

        elapsed = time.time() - start_time
        print(f"Elapsed hashing time : {elapsed} sec")
        list_time.append(elapsed)

        print("Save result for final Json ... ")
        json_add_top_matches(sorted_picture_list, curr_target_picture)

    elapsed_FULL_TIME = time.time() - start_FULL_time
    print(f"Elapsed hashing time : {elapsed_FULL_TIME} sec for {len(picture_list)} items ({elapsed_FULL_TIME / len(picture_list)} per item)")

    print("Export json ... ")
    json_export(json_to_export, "test.json")

    print_stats(stats.describe(list_time))


ROUND_DECIMAL = 5


def print_stats(stats_result):
    tmp_str = ""
    tmp_str += "nobs : " + str(getattr(stats_result, "nobs")) + " "
    tmp_str += "min time : " + str(round(getattr(stats_result, "minmax")[0], ROUND_DECIMAL)) + "s "
    tmp_str += "max time : " + str(round(getattr(stats_result, "minmax")[1], ROUND_DECIMAL)) + "s "
    tmp_str += "mean :" + str(round(getattr(stats_result, "mean"), ROUND_DECIMAL)) + "s "
    tmp_str += "variance : " + str(round(getattr(stats_result, "variance"), ROUND_DECIMAL)) + "s "
    tmp_str += "skewness : " + str(round(getattr(stats_result, "skewness"), ROUND_DECIMAL)) + "s "
    tmp_str += "kurtosis : " + str(round(getattr(stats_result, "kurtosis"), ROUND_DECIMAL))
    print(tmp_str)


def clean_folder(target_dir):
    '''
    Remove 0-bytes size files
    :param target_dir:
    :return:
    '''

    pathlist = pathlib.Path(target_dir).glob('**/*.png')
    for i, path in enumerate(pathlist):
        if path.stat().st_size == 0:
            path.unlink()


if __name__ == '__main__':
    target_dir = "../../datasets/raw_phishing/"
    clean_folder(target_dir)
    # random_test(target_dir)
    full_test(target_dir)
