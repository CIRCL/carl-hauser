import time
import pathlib
from scipy import stats

from utility_lib import filesystem_lib
from utility_lib import printing_lib
from utility_lib import stats_lib
from utility_lib import picture_class
from utility_lib import json_class

DEFAULT_TARGET_DIR = "../../datasets/raw_phishing/"
DEFAULT_OUTPUT_DIR = "./RESULTS/"

class Execution_handler() :
    def __init__(self, target_dir=DEFAULT_TARGET_DIR, Local_Picture=picture_class.Picture, save_picture=False, output_dir=DEFAULT_OUTPUT_DIR):
        self.target_dir = target_dir
        self.output_dir = output_dir
        self.pathlist = pathlib.Path(target_dir).glob('**/*.png')
        self.Local_Picture_class_ref = Local_Picture
        self.save_picture = save_picture

        # Used during process
        self.target_picture = None
        self.picture_list = []
        self.sorted_picture_list = []
        self.JSON_file_object = json_class.JSON_VISUALISATION()
        # Multipurpose storage, e.g. store some useful class for processing.
        self.storage = None

        # For statistics only
        self.list_time = []

        # Actions handlers
        self.printer = printing_lib.Printer()

    def do_random_test(self):
        print("=============== RANDOM TEST SELECTED ===============")
        self.pick_random_picture_handler()
        self.load_pictures()
        self.prepare_dataset()
        self.prepare_target_picture()
        self.find_closest_picture()
        self.find_top_k_closest_pictures()
        self.save_pictures()

    def do_full_test(self):
        print("=============== FULL TEST SELECTED ===============")
        self.load_pictures()
        self.prepare_initial_JSON()
        self.prepare_dataset()
        self.iterate_over_dataset()
        self.export_final_JSON()
        self.describe_stats()

    # ====================== STEPS OF ALGORITHMS ======================

    def pick_random_picture_handler(self):
        print("Pick a random picture ... ")
        target_picture_path = filesystem_lib.random_choice(self.target_dir)
        self.target_picture = self.Local_Picture_class_ref(id=None, path=target_picture_path)
        print("Target picture : " + str(target_picture_path))

    def load_pictures(self):
        print("Load pictures ... ")
        self.picture_list = filesystem_lib.get_Pictures_from_directory(self.target_dir, class_name=self.Local_Picture_class_ref)

    def prepare_dataset(self):
        print("Prepare dataset pictures ... (Launch timer)")
        start_time = time.time()
        self.TO_OVERWRITE_prepare_dataset()
        self.print_elapsed_time(time.time() - start_time, len(self.picture_list))

    def TO_OVERWRITE_prepare_dataset(self):
        print("PREPARE_DATASET HASN'T BEEN OVERWRITE. PLEASE DO OVERWRITE PARENT FUNCTION BEFORE LAUNCH")


    def prepare_target_picture(self):
        print("Prepare target picture ... (Launch timer)")
        start_time = time.time()
        self.TO_OVERWRITE_prepare_target_picture()
        self.print_elapsed_time(time.time() - start_time, len(self.picture_list))

    def TO_OVERWRITE_prepare_target_picture(self):
        print("PREPARE_DATASET HASN'T BEEN OVERWRITE. PLEASE DO OVERWRITE PARENT FUNCTION BEFORE LAUNCH")

    def iterate_over_dataset(self):
        print("Iterate over dataset ... (Launch global timer)")
        start_FULL_time = time.time()
        for i, curr_target_picture in enumerate(self.picture_list):
            print(f"PICTURE {i} picked as target ... (start current timer)")
            print("Target picture : " + str(curr_target_picture.path))
            start_time = time.time()

            self.target_picture = curr_target_picture
            try :
                self.find_closest_picture()
                self.find_top_k_closest_pictures()
                self.save_pictures()
                self.add_top_matches_to_JSON()
            except Exception as e :
                print(f"An Exception has occured during the tentative to find a match to {curr_target_picture.path.name}")

            elapsed = time.time() - start_time
            self.print_elapsed_time(elapsed, 1, to_add="current ")
            self.list_time.append(elapsed)

        self.print_elapsed_time(time.time() - start_FULL_time, len(self.picture_list), to_add="global ")

    def find_closest_picture(self):
        print("Find closest picture from target picture ... ")
        picture_class.find_closest(self.picture_list, self.target_picture) # TODO : To remove ? Not useful ?

    def find_top_k_closest_pictures(self):
        print("Extract top K images ... ")
        self.picture_list = [i for i in self.picture_list if i.distance is not None]
        # TODO : Remove None values : TO CHECK WHAT WE CAN DO !
        self.sorted_picture_list = picture_class.get_top(self.picture_list, self.target_picture)

    def save_pictures(self):
        if self.save_picture:
            self.printer.save_picture_top_matches(self.sorted_picture_list, self.target_picture, DEFAULT_OUTPUT_DIR + self.target_picture.path.name)

    # ====================== JSON HANDLING ======================

    def prepare_initial_JSON(self):
        self.JSON_file_object.json_add_nodes(self.picture_list)

    def add_top_matches_to_JSON(self):
        print("Save result for final Json ... ")
        self.JSON_file_object.json_add_top_matches(self.sorted_picture_list, self.target_picture)

    def export_final_JSON(self):
        print("Export json ... ")
        self.JSON_file_object.json_export("test.json")

    # ====================== STATISTICS AND PRINTING ======================

    def describe_stats(self):
        print("Describing timer statistics... ")
        stats_lib.print_stats(stats.describe(self.list_time))

    @staticmethod
    def print_elapsed_time(elapsed_time, nb_item, to_add=""):

        E1 = round(elapsed_time,stats_lib.ROUND_DECIMAL)
        E2 = round(elapsed_time/nb_item,stats_lib.ROUND_DECIMAL)

        print(f"Elapsed computation {to_add}time : {E1}s for {nb_item} items ({E2}s per item)")
