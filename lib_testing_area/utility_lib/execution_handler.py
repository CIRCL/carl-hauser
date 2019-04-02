import time
import pathlib
import operator
from scipy import stats

from utility_lib import filesystem_lib
from utility_lib import printing_lib
from utility_lib import stats_lib
from utility_lib import picture_class
from utility_lib import json_class

import configuration
import logging
import pprint

class Execution_handler():
    def __init__(self, conf: configuration.Default_configuration):
        print("Initialisation of Execution handler ...")
        print("Actual configuration : ")
        pprint.pprint(vars(conf))
        self.conf = conf

        # Actions handlers
        self.printer = printing_lib.Printer(conf=self.conf)
        self.file_system = filesystem_lib.File_System(conf=self.conf)
        self.JSON_graphe_object = json_class.JSON_GRAPHE()
        self.logger = logging.getLogger(__name__)

        # For statistics only
        self.list_time = [] #TODO : To replace

        # Load primary configuration parameters
        # TODO : Checks on the dir to verify they exist
        self.conf.SOURCE_DIR = self.file_system.safe_path(conf.SOURCE_DIR)
        self.conf.OUTPUT_DIR = self.file_system.safe_path(conf.OUTPUT_DIR)

        if self.conf.IMG_TYPE == configuration.SUPPORTED_IMAGE_TYPE.PNG:
            self.pathlist = self.conf.SOURCE_DIR.glob('**/*.png')
        elif self.conf.IMG_TYPE == configuration.SUPPORTED_IMAGE_TYPE.BMP:
            self.pathlist = self.conf.SOURCE_DIR.glob('**/*.bmp')
        else:
            raise Exception("IMG_TYPE not recognized as a valid type" + str(self.conf.IMG_TYPE.value))

        # Default Local_picture that has to be overwrite
        self.Local_Picture_class_ref = picture_class.Picture
        self.save_picture = self.conf.SAVE_PICTURE

        # Used during process
        self.target_picture = None
        self.picture_list = []
        self.sorted_picture_list = []


    def do_random_test(self):
        print("=============== RANDOM TEST SELECTED ===============")
        self.target_picture = self.pick_random_picture_handler(self.conf.SOURCE_DIR)
        self.picture_list = self.load_pictures(self.conf.SOURCE_DIR, self.Local_Picture_class_ref)
        self.picture_list = self.prepare_dataset(self.picture_list)
        self.target_picture = self.prepare_target_picture(self.target_picture)
        # self.find_closest_picture(self.picture_list, self.target_picture)
        self.sorted_picture_list = self.find_top_k_closest_pictures(self.picture_list, self.conf.SOURCE_DIR)
        self.save_pictures(self.sorted_picture_list, self.target_picture)

    def do_full_test(self):
        print("=============== FULL TEST SELECTED ===============")
        self.picture_list = self.load_pictures(self.conf.SOURCE_DIR, self.Local_Picture_class_ref)
        self.JSON_graphe_object = self.prepare_initial_JSON(self.picture_list, self.JSON_graphe_object)
        self.picture_list = self.prepare_dataset(self.picture_list)
        self.JSON_graphe_object, self.list_time = self.iterate_over_dataset(self.picture_list, self.JSON_graphe_object)
        self.JSON_graphe_object = self.evaluate_JSON(self.JSON_graphe_object, self.conf.GROUND_TRUTH_PATH)
        self.export_final_JSON(self.JSON_graphe_object)
        self.describe_stats(self.list_time)

    # ====================== STEPS OF ALGORITHMS ======================

    def pick_random_picture_handler(self, target_dir : pathlib.Path):
        self.logger.info("Pick a random picture ... ")
        target_picture_path = self.file_system.random_choice(target_dir)
        self.logger.info(f"Target picture : {target_picture_path}")

        target_picture = self.Local_Picture_class_ref(id=None, conf=self.conf, path=target_picture_path)
        return target_picture

    def load_pictures(self, target_dir : pathlib.Path, Local_Picture_class_ref):
        print("Load pictures ... ")
        picture_list = self.file_system.get_Pictures_from_directory(target_dir, class_name=Local_Picture_class_ref)
        return picture_list

    def prepare_dataset(self, picture_list):
        print("Prepare dataset pictures ... (Launch timer)")
        start_time = time.time()
        picture_list = self.TO_OVERWRITE_prepare_dataset(picture_list)
        self.print_elapsed_time(time.time() - start_time, len(picture_list))
        return picture_list

    def TO_OVERWRITE_prepare_dataset(self, picture_list):
        raise Exception("PREPARE_DATASET HASN'T BEEN OVERWRITE. PLEASE DO OVERWRITE PARENT FUNCTION BEFORE LAUNCH")
        return picture_list

    def prepare_target_picture(self, target_picture):
        print("Prepare target picture ... (Launch timer)")
        start_time = time.time()
        target_picture = self.TO_OVERWRITE_prepare_target_picture(target_picture)
        self.print_elapsed_time(time.time() - start_time, 1)
        return target_picture

    def TO_OVERWRITE_prepare_target_picture(self, target_picture):
        raise Exception("PREPARE_DATASET HASN'T BEEN OVERWRITE. PLEASE DO OVERWRITE PARENT FUNCTION BEFORE LAUNCH")
        return target_picture

    def iterate_over_dataset(self, picture_list, JSON_file_object):
        print("Iterate over dataset ... (Launch global timer)")

        if len(picture_list) == 0 or picture_list == []:
            raise Exception("ITERATE OVER DATASET IN EXECUTION HANDLER : Picture list empty ! Abort.")

        list_time = []
        start_FULL_time = time.time()
        for i, curr_target_picture in enumerate(picture_list):
            print(f"PICTURE {i} picked as target ... (start current timer)")
            print("Target picture : " + str(curr_target_picture.path))
            start_time = time.time()

            if curr_target_picture.path.name == "hosterfalr.ga.png":
                print("found")
            try:
                # self.find_closest_picture(picture_list, curr_target_picture)
                curr_sorted_picture_list = self.find_top_k_closest_pictures(picture_list, curr_target_picture)
            except Exception as e:
                print(f"An Exception has occured during the tentative to find a (k-top) match to {curr_target_picture.path.name} : " + str(e))
                raise e

            try:
                self.save_pictures(curr_sorted_picture_list, curr_target_picture)
            except Exception as e:
                print(f"An Exception has occured during the tentative save the result picture of {curr_target_picture.path.name} : " + str(e))
                raise e

            try:
                # if curr_sorted_picture_list[0].distance < THREESHOLD :
                JSON_file_object = self.add_top_matches_to_JSON(curr_sorted_picture_list, curr_target_picture, JSON_file_object)
            except Exception as e:
                print(f"An Exception has occured during the tentative to add result to json for {curr_target_picture.path.name} : " + str(e))
                raise e

            elapsed = time.time() - start_time
            self.print_elapsed_time(elapsed, 1, to_add="current ")
            list_time.append(elapsed)

        self.print_elapsed_time(time.time() - start_FULL_time, len(picture_list), to_add="global ")
        return JSON_file_object, list_time

    def find_closest_picture(self, picture_list, target_picture):
        # TODO : To remove ? Not useful ?
        print("Find closest picture from target picture ... ")
        if picture_list == None or picture_list == []:
            raise Exception("PICTURE_CLASS : Provided picture list is empty.")
        if target_picture == None:
            raise Exception("PICTURE_CLASS : Provided target picture is empty.")

        min = None
        min_object = None

        for curr_picture in picture_list:
            curr_picture.distance = self.TO_OVERWRITE_compute_distance(curr_picture, target_picture)
            curr_dist = curr_picture.dist

            if not target_picture.is_same_picture_as(curr_picture) and curr_dist is not None and (min is None or min > curr_dist):
                min = curr_dist
                min_object = curr_picture

        if min is None or min_object is None:
            print("No best match found. No picture seems to have even a big distance from the target.")
            raise Exception("PICTURE_CLASS : No object found at a minimal distance. Most likely a library error or a too narrow dataset.")
        else:
            print("original picture : \t" + str(target_picture.path))
            print("min found : \t" + str(min_object.path) + " with " + str(min))

    def find_top_k_closest_pictures(self, picture_list, target_picture):
        # Compute distances
        for curr_pic in picture_list:
            curr_pic.distance = self.TO_OVERWRITE_compute_distance(curr_pic, target_picture)

        print("Extract top K images ... ")
        picture_list = [i for i in picture_list if i.distance is not None]
        print(f"Candidate picture list length : {len(picture_list)}")

        sorted_picture_list = self.get_top(picture_list, target_picture)
        return sorted_picture_list

    def get_top(self, picture_list, target_picture):
        for curr_picture in picture_list:
            curr_picture.distance = self.TO_OVERWRITE_compute_distance(curr_picture, target_picture)

        # Remove None distance
        picture_list = [item for item in picture_list if item.distance != None]

        sorted_picture_list = sorted(picture_list, key=operator.attrgetter('distance'))
        # print(sorted_picture_list)

        return sorted_picture_list

    def TO_OVERWRITE_compute_distance(self, pic1, pic2):
        raise Exception("COMPUTE_DISTANCE_EXT HASN'T BEEN OVERWRITE. PLEASE DO OVERWRITE PARENT FUNCTION BEFORE LAUNCH")
        return None

    def save_pictures(self, sorted_picture_list, target_picture):
        if self.save_picture:
            # TODO : GENERATE A GOOD NAME
            self.printer.save_picture_top_matches(sorted_picture_list, target_picture, self.conf.OUTPUT_DIR / "TEST"  / target_picture.path.name)

    @staticmethod
    def print_list(list, threeshold= 5):
        for i in list[0:threeshold]:
            print(str(i.path) + " : " + str(i.distance))

    # ====================== JSON HANDLING ======================

    def prepare_initial_JSON(self, picture_list, JSON_file_object):
        return JSON_file_object.json_add_nodes(picture_list)

    def add_top_matches_to_JSON(self, sorted_picture_list, target_picture, tmp_json_object):
        print("Save result for final Json ... ")
        tmp_json_object.json_to_export = tmp_json_object.json_add_top_matches(tmp_json_object.json_to_export, sorted_picture_list, target_picture)
        return tmp_json_object

    def export_final_JSON(self, JSON_file_object):
        print("Export json ... ")
        JSON_file_object.json_export("test.json")

    # ====================== STATISTICS AND PRINTING ======================
    def evaluate_JSON(self, JSON_file_object, baseline_path):
        JSON_file_object, quality = JSON_file_object.evaluate_json(pathlib.Path(baseline_path))
        print(f"Quality of guess : {str(round(quality, stats_lib.ROUND_DECIMAL))}")
        return JSON_file_object

    def describe_stats(self, list_time):
        print("Describing timer statistics... ")
        stats_lib.print_stats(stats.describe(list_time))

    @staticmethod
    def print_elapsed_time(elapsed_time, nb_item, to_add=""):

        if nb_item == 0:
            print("Print elapsed time : ERROR - nb_item = 0")
            nb_item = 1
        E1 = round(elapsed_time, stats_lib.ROUND_DECIMAL)
        E2 = round(elapsed_time / nb_item, stats_lib.ROUND_DECIMAL)

        print(f"Elapsed computation {to_add}time : {E1}s for {nb_item} items ({E2}s per item)")
