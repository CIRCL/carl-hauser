from typing import List
import operator
from PIL import Image

THREESHOLD = 5

class Picture() :
    def __init__(self, id, shape="image", path=None):
        self.id = id
        self.shape = shape
        self.path = path
        self.matched = False
        self.sorted_matching_picture_list = []
        # Hashing related attributes
        self.hash = None
        self.distance = None
        # Descriptors related attributes
        self.key_points = None
        self.description = None
        self.image = self.load_image(self.path)
        # Multipurpose storage, e.g. store some useful class for processing.
        self.storage = None
        self.matches = None

    def load_image(self, path):
        if path is None or path == "" :
            return None

        return Image.open(str(path))

    def is_same_picture_as(self, pic1):
        # TODO : Except on SHA1 hash ?
        return self.path == pic1.path

    def to_node_json_object(self):
        tmp_obj = {}
        tmp_obj["id"] = self.id
        tmp_obj["shape"] = self.shape
        tmp_obj["image"] = self.path.name
        return tmp_obj

    def compute_distance(self, target_picture):
        self.distance = self.compute_distance_ext(self, target_picture)
        return self.distance

    def compute_distance_ext(self, pic1, pic2):
        raise Exception("COMPUTE_DISTANCE_EXT HASN'T BEEN OVERWRITE. PLEASE DO OVERWRITE PARENT FUNCTION BEFORE LAUNCH")
        return None

    ''' OVERWRITE EXAMPLE :
            self.distance = abs(pic1.hash - pic2.hash)
        return self.distance
    '''

def find_closest(picture_list: List[Picture], target_picture: Picture):
    if picture_list == None or picture_list == [] :
        raise Exception("PICTURE_CLASS : Provided picture list is empty.")
    if target_picture == None :
        raise Exception("PICTURE_CLASS : Provided target picture is empty.")

    min = None
    min_object = None

    for curr_picture in picture_list:
        curr_picture.compute_distance(target_picture)
        curr_dist = target_picture.distance

        if not target_picture.is_same_picture_as(curr_picture) and curr_dist is not None and (min is None or min > curr_dist):
            min = curr_dist
            min_object = curr_picture

    if min is None or min_object is None :
        print("No best match found. No picture seems to have even a big distance from the target.")
        raise Exception("PICTURE_CLASS : No object found at a minimal distance. Most likely a library error or a too narrow dataset.")
    else :
        print("original picture : \t" + str(target_picture.path))
        print("min found : \t" + str(min_object.path) + " with " + str(min))

def get_top(picture_list: List[Picture], target_picture: Picture):
    for curr_picture in picture_list:
        curr_picture.compute_distance(target_picture)

    # Remove None distance
    picture_list = [item for item in picture_list if item.distance != None]

    sorted_picture_list = sorted(picture_list, key=operator.attrgetter('distance'))
    # print(sorted_picture_list)

    return sorted_picture_list


def print_list(list, threeshold=THREESHOLD):
    for i in list[0:THREESHOLD]:
        print(str(i.path) + " : " + str(i.distance))

