import time
import pathlib
import operator

from utility_lib import filesystem_lib
from utility_lib import printing_lib
from utility_lib import stats_lib
from utility_lib import picture_class
from utility_lib import json_class

import configuration
import results
import logging
import pprint

class Execution_handler():
    def __init__(self, conf: configuration.Default_configuration):
        self.conf = conf
        self.results_storage = results.RESULTS()

        logging.basicConfig(filename=str(conf.OUTPUT_DIR/"execution.log"), filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.logger.info("Initialisation of Execution handler ...")

        # Load primary configuration parameters
        # TODO : Checks on the dir to verify they exist
        self.file_system = filesystem_lib.File_System(conf=self.conf)
        self.conf.SOURCE_DIR = self.file_system.safe_path(conf.SOURCE_DIR)
        self.conf.OUTPUT_DIR = self.file_system.safe_path(conf.OUTPUT_DIR)
        # Passed by reference ? No need ? # self.file_system.conf = self.conf

        self.logger.debug("Actual configuration : ")
        self.logger.debug(pprint.pformat(vars(conf)))

        # Actions handlers
        self.printer = printing_lib.Printer(conf=self.conf)
        self.stats_handler = stats_lib.Stats_handler(conf=self.conf)
        self.json_handler = json_class.Json_handler(conf=self.conf)

        self.create_folder(self.conf.OUTPUT_DIR)
        self.write_configuration_to_folder(self.conf)

        # For statistics only
        self.list_time = []  # TODO : To replace

        if self.conf.IMG_TYPE == configuration.SUPPORTED_IMAGE_TYPE.PNG:
            self.pathlist = self.conf.SOURCE_DIR.glob('**/*.png')
        elif self.conf.IMG_TYPE == configuration.SUPPORTED_IMAGE_TYPE.BMP:
            self.pathlist = self.conf.SOURCE_DIR.glob('**/*.bmp')
        else:
            raise Exception(f"IMG_TYPE not recognized as a valid type {self.conf.IMG_TYPE.value}")

        # Default Local_picture that has to be overwrite
        self.Local_Picture_class_ref = picture_class.Picture
        self.save_picture = self.conf.SAVE_PICTURE

        # Used during process
        self.target_picture = None
        self.picture_list = []
        self.sorted_picture_list = []

    def do_random_test(self):
        self.logger.info("=============== RANDOM TEST SELECTED ===============")
        self.target_picture = self.pick_random_picture_handler(self.conf.SOURCE_DIR)
        self.picture_list = self.load_pictures(self.conf.SOURCE_DIR, self.Local_Picture_class_ref)
        self.picture_list = self.prepare_dataset(self.picture_list)
        self.target_picture = self.prepare_target_picture(self.target_picture)
        # self.find_closest_picture(self.picture_list, self.target_picture)
        self.sorted_picture_list = self.find_top_k_closest_pictures(self.picture_list, self.conf.SOURCE_DIR)
        self.save_pictures(self.sorted_picture_list, self.target_picture)

    def do_full_test(self):
        self.logger.info("=============== FULL TEST SELECTED ===============")
        self.picture_list = self.load_pictures(self.conf.SOURCE_DIR, self.Local_Picture_class_ref)
        self.json_handler = self.prepare_initial_JSON(self.picture_list, self.json_handler)
        self.picture_list = self.prepare_dataset(self.picture_list)
        self.json_handler, self.list_time = self.iterate_over_dataset(self.picture_list, self.json_handler)
        self.json_handler = self.evaluate_JSON(self.json_handler, self.conf.GROUND_TRUTH_PATH)
        self.export_final_JSON(self.json_handler)
        self.describe_stats(self.list_time)

    # ====================== STEPS OF ALGORITHMS ======================

    def pick_random_picture_handler(self, target_dir: pathlib.Path):
        self.logger.info("Pick a random picture ... ")
        target_picture_path = self.file_system.random_choice(target_dir)
        self.logger.info(f"Target picture : {target_picture_path}")

        target_picture = self.Local_Picture_class_ref(id=None, conf=self.conf, path=target_picture_path)
        return target_picture

    def load_pictures(self, target_dir: pathlib.Path, Local_Picture_class_ref):
        self.logger.info("Load pictures ... ")
        start_time = time.time()
        picture_list = self.file_system.get_Pictures_from_directory(target_dir, class_name=Local_Picture_class_ref)

        self.results_storage.TIME_TO_LOAD_PICTURES = time.time() - start_time
        self.print_elapsed_time(self.results_storage.TIME_TO_LOAD_PICTURES, 1)
        return picture_list

    def prepare_dataset(self, picture_list):
        self.logger.info("Prepare dataset pictures ... (Launch timer)")
        start_time = time.time()
        picture_list = self.TO_OVERWRITE_prepare_dataset(picture_list)

        self.results_storage.TIME_TOTAL_PRE_COMPUTING = time.time() - start_time
        self.results_storage.TIME_PER_PICTURE_PRE_COMPUTING = self.results_storage.TIME_TOTAL_PRE_COMPUTING / len(picture_list)

        self.print_elapsed_time(self.results_storage.TIME_TOTAL_PRE_COMPUTING, len(picture_list))
        return picture_list

    def TO_OVERWRITE_prepare_dataset(self, picture_list):
        raise Exception("PREPARE_DATASET HASN'T BEEN OVERWRITE. PLEASE DO OVERWRITE PARENT FUNCTION BEFORE LAUNCH")
        return picture_list

    def prepare_target_picture(self, target_picture):
        self.logger.info("Prepare target picture ... (Launch timer)")
        start_time = time.time()
        target_picture = self.TO_OVERWRITE_prepare_target_picture(target_picture)

        self.results_storage.TIME_REQUEST_PICTURE_COMPUTING = time.time() - start_time
        self.print_elapsed_time(self.results_storage.TIME_REQUEST_PICTURE_COMPUTING, 1)
        return target_picture

    def TO_OVERWRITE_prepare_target_picture(self, target_picture):
        raise Exception("PREPARE_DATASET HASN'T BEEN OVERWRITE. PLEASE DO OVERWRITE PARENT FUNCTION BEFORE LAUNCH")
        return target_picture

    def iterate_over_dataset(self, picture_list, json_handler):
        self.logger.info("Iterate over dataset ... (Launch global timer)")

        if len(picture_list) == 0 or picture_list == []:
            raise Exception("ITERATE OVER DATASET IN EXECUTION HANDLER : Picture list empty ! Abort.")

        list_time = []
        start_FULL_time = time.time()
        for i, curr_target_picture in enumerate(picture_list):
            self.logger.info(f"PICTURE {i} picked as target ... (start current timer)")
            self.logger.debug(f"Target picture : {curr_target_picture.path}")
            start_time = time.time()

            try:
                # self.find_closest_picture(picture_list, curr_target_picture)
                curr_sorted_picture_list = self.find_top_k_closest_pictures(picture_list, curr_target_picture)
            except Exception as e:
                self.logger.error(
                    f"An Exception has occured during the tentative to find a (k-top) match to {curr_target_picture.path.name} : " + str(e))
                raise e

            try:
                self.save_pictures(curr_sorted_picture_list, curr_target_picture)
            except Exception as e:
                self.logger.error(
                    f"An Exception has occured during the tentative save the result picture of {curr_target_picture.path.name} : " + str(e))
                raise e

            try:
                # if curr_sorted_picture_list[0].distance < THREESHOLD :
                json_handler = self.add_top_matches_to_JSON(curr_sorted_picture_list, curr_target_picture, json_handler)
            except Exception as e:
                self.logger.error(
                    f"An Exception has occured during the tentative to add result to json for {curr_target_picture.path.name} : " + str(e))
                raise e

            elapsed = time.time() - start_time
            self.print_elapsed_time(elapsed, 1, to_add="current ")
            list_time.append(elapsed)

        self.results_storage.TIME_TOTAL_MATCHING = time.time() - start_FULL_time
        self.results_storage.TIME_LIST_MATCHING = list_time
        self.results_storage.NB_PICTURE = len(picture_list)
        self.results_storage.TIME_PER_PICTURE_MATCHING = self.results_storage.TIME_TOTAL_MATCHING / len(picture_list)

        self.print_elapsed_time(self.results_storage.TIME_TOTAL_MATCHING, len(picture_list), to_add="global ")
        return json_handler, list_time

    def find_closest_picture(self, picture_list, target_picture):
        # TODO : To remove ? Not useful ?
        self.logger.info("Find closest picture from target picture ... ")
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
            self.logger.error("No best match found. No picture seems to have even a big distance from the target.")
            raise Exception("PICTURE_CLASS : No object found at a minimal distance. Most likely a library error or a too narrow dataset.")
        else:
            self.logger.debug("original picture : \t" + str(target_picture.path))
            self.logger.debug("min found : \t" + str(min_object.path) + " with " + str(min))

    def find_top_k_closest_pictures(self, picture_list, target_picture):
        # Compute distances
        for curr_pic in picture_list:
            curr_pic.distance = self.TO_OVERWRITE_compute_distance(curr_pic, target_picture)

        self.logger.info("Extract top K images ... ")
        picture_list = [i for i in picture_list if i.distance is not None]
        self.logger.debug(f"Candidate picture list length : {len(picture_list)}")

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
            self.printer.save_picture_top_matches(sorted_picture_list, target_picture, self.conf.OUTPUT_DIR / target_picture.path.name)

    @staticmethod
    def print_list(list, threeshold=5):
        logger = logging.getLogger(__name__)

        for i in list[0:threeshold]:
            logger.debug(str(i.path) + " : " + str(i.distance))

    # ====================== JSON HANDLING ======================

    def prepare_initial_JSON(self, picture_list, json_handler):
        return json_handler.json_add_nodes(picture_list)

    def add_top_matches_to_JSON(self, sorted_picture_list, target_picture, json_handler):
        self.logger.info("Save result for final Json ... ")
        json_handler.json_to_export = json_handler.json_add_top_matches(json_handler.json_to_export, sorted_picture_list, target_picture)
        return json_handler

    def export_final_JSON(self, json_handler):
        self.logger.info("Export json ... ")
        json_handler.json_export()

    # ====================== STATISTICS AND PRINTING ======================
    def evaluate_JSON(self, json_handler, baseline_path):
        json_handler, self.results_storage = json_handler.evaluate_json(pathlib.Path(baseline_path), self.results_storage)
        self.logger.info(f"Quality of guess : {str(round(self.results_storage.TRUE_POSITIVE_RATE, stats_lib.ROUND_DECIMAL))}")
        return json_handler

    def describe_stats(self, list_time):
        self.logger.info("Describing timer statistics... ")
        self.stats_handler.print_stats(self.conf, self.results_storage)
        self.stats_handler.write_stats_to_folder(self.conf, self.results_storage)

    @staticmethod
    def print_elapsed_time(elapsed_time, nb_item, to_add=""):
        logger = logging.getLogger(__name__)

        if nb_item == 0:
            logger.info("Print elapsed time : ERROR - nb_item = 0")
            nb_item = 1
        E1 = round(elapsed_time, stats_lib.ROUND_DECIMAL)
        E2 = round(elapsed_time / nb_item, stats_lib.ROUND_DECIMAL)

        logger.info(f"Elapsed computation {to_add}time : {E1}s for {nb_item} items ({E2}s per item)")

    @staticmethod
    def conf_to_string(conf: configuration.Default_configuration):
        answer = ""
        final_char = "_"

        answer += conf.SOURCE_DIR.name
        answer += final_char + conf.IMG_TYPE.name
        answer += final_char + conf.ALGO.name

        if conf.SELECTION_THREESHOLD is not None:
            answer += final_char + "THREE_" + str(conf.SELECTION_THREESHOLD)

        if type(conf) == configuration.ORB_default_configuration:
            answer += final_char + str(conf.ORB_KEYPOINTS_NB)
            answer += final_char + conf.DISTANCE.name
            answer += final_char + conf.FILTER.name
            answer += final_char + conf.MATCH.name
            if conf.MATCH == configuration.MATCH_TYPE.KNN:
                answer += final_char + str(conf.MATCH_K_FOR_KNN)
            answer += final_char + conf.DATASTRUCT.name
            answer += final_char + conf.CROSSCHECK.name

        return answer

    def create_folder(self, path: pathlib.PosixPath):
        path.mkdir(parents=True, exist_ok=True)

        self.logger.debug(f"Folder {path} created.")

    def write_configuration_to_folder(self, conf: configuration.Default_configuration):
        fn = "conf.txt"
        filepath = conf.OUTPUT_DIR / fn
        with filepath.open("w", encoding="utf-8") as f:
            f.write(pprint.pformat(vars(conf)))

        self.logger.debug(f"Configuration file saved as {filepath}.")
