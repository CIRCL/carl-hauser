import time
import pathlib
from scipy import stats
from PIL import Image
from enum import Enum, auto

from utility_lib import filesystem_lib
from utility_lib import printing_lib
from utility_lib import stats_lib
from utility_lib import picture_class
from utility_lib import json_class


class SOURCE(Enum):
    raw_phishing_png = auto()
    raw_phishing_bmp = auto()
SOURCE_CHOSEN = SOURCE.raw_phishing_png

if SOURCE_CHOSEN == SOURCE.raw_phishing_bmp :
    DEFAULT_TARGET_DIR = "../../datasets/raw_phishing_bmp/"
    DEFAULT_BASELINE_PATH = "../../datasets/raw_phishing_bmp.json"
    FILE_TYPE = ".bmp"
elif SOURCE_CHOSEN == SOURCE.raw_phishing_png :
    DEFAULT_TARGET_DIR = "../../datasets/raw_phishing/"
    DEFAULT_BASELINE_PATH = "../../datasets/raw_phishing.json"
    FILE_TYPE = ".png"

DEFAULT_OUTPUT_DIR = "./RESULTS/"

THREESHOLD = 0.7

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
        self.file_system = filesystem_lib.File_System(type=FILE_TYPE)

    def do_random_test(self):
        print("=============== RANDOM TEST SELECTED ===============")
        self.target_picture = self.pick_random_picture_handler(self.target_dir)
        self.picture_list = self.load_pictures(self.target_dir, self.Local_Picture_class_ref)
        self.picture_list = self.prepare_dataset(self.picture_list)
        self.target_picture = self.prepare_target_picture(self.target_picture)
        # self.find_closest_picture(self.picture_list, self.target_picture)
        self.sorted_picture_list = self.find_top_k_closest_pictures(self.picture_list, self.target_dir)
        self.save_pictures(self.sorted_picture_list, self.target_picture)

    def do_full_test(self):
        print("=============== FULL TEST SELECTED ===============")
        self.picture_list = self.load_pictures(self.target_dir, self.Local_Picture_class_ref)
        self.JSON_file_object = self.prepare_initial_JSON(self.picture_list, self.JSON_file_object)
        self.picture_list = self.prepare_dataset(self.picture_list)
        self.JSON_file_object, self.list_time = self.iterate_over_dataset(self.picture_list, self.JSON_file_object)
        self.JSON_file_object = self.evaluate_JSON(self.JSON_file_object)
        self.export_final_JSON(self.JSON_file_object)
        self.describe_stats(self.list_time)

    # ====================== STEPS OF ALGORITHMS ======================

    def pick_random_picture_handler(self, target_dir):
        print("Pick a random picture ... ")
        target_picture_path = self.file_system.random_choice(target_dir)
        print("Target picture : " + str(target_picture_path))

        target_picture = self.Local_Picture_class_ref(id=None, path=target_picture_path)
        return target_picture

    def load_pictures(self, target_dir, Local_Picture_class_ref):
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

            curr_sorted_picture_list = []

            if curr_target_picture.path.name == "hosterfalr.ga.png":
                print("found")

            try :
                # self.find_closest_picture(picture_list, curr_target_picture)
                curr_sorted_picture_list = self.find_top_k_closest_pictures(picture_list, curr_target_picture)
            except Exception as e :
                print(f"An Exception has occured during the tentative to find a (k-top) match to {curr_target_picture.path.name} : " + str(e))
                raise e
            try :
                self.save_pictures(curr_sorted_picture_list, curr_target_picture)
            except Exception as e :
                print(f"An Exception has occured during the tentative save the result picture of {curr_target_picture.path.name} : " + str(e))
                raise e

            try :
                # if curr_sorted_picture_list[0].distance < THREESHOLD :
                JSON_file_object = self.add_top_matches_to_JSON(curr_sorted_picture_list, curr_target_picture, JSON_file_object)
            except Exception as e :
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
        picture_class.find_closest(picture_list, target_picture) # TODO : return

    def find_top_k_closest_pictures(self, picture_list, target_picture):
        # Compute distances
        for curr_pic in picture_list:
            curr_pic.compute_distance(target_picture)

        print("Extract top K images ... ")
        picture_list = [i for i in picture_list if i.distance is not None]
        print(f"Candidate picture list length : {len(picture_list)}")

        sorted_picture_list = picture_class.get_top(picture_list, target_picture)
        return sorted_picture_list

    def save_pictures(self, sorted_picture_list, target_picture):
        if self.save_picture:
            self.printer.save_picture_top_matches(sorted_picture_list, target_picture, DEFAULT_OUTPUT_DIR + target_picture.path.name)

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
    def evaluate_JSON(self, JSON_file_object, baseline_path =DEFAULT_BASELINE_PATH):
        JSON_file_object, quality = JSON_file_object.evaluate_json(pathlib.Path(baseline_path))
        print(f"Quality of guess : {str(round(quality,stats_lib.ROUND_DECIMAL))}")
        return JSON_file_object

    def describe_stats(self, list_time):
        print("Describing timer statistics... ")
        stats_lib.print_stats(stats.describe(list_time))

    @staticmethod
    def print_elapsed_time(elapsed_time, nb_item, to_add=""):

        if nb_item == 0 :
            print("Print elapsed time : ERROR - nb_item = 0")
            nb_item = 1
        E1 = round(elapsed_time,stats_lib.ROUND_DECIMAL)
        E2 = round(elapsed_time/nb_item,stats_lib.ROUND_DECIMAL)

        print(f"Elapsed computation {to_add}time : {E1}s for {nb_item} items ({E2}s per item)")



# ============================================= ------------------------------------ =============================================
#                                                           MANUAL LABOR

DEFAULT_BASELINE_PATH = "../../datasets/raw_phishing.json"
BMP_TARGET_DIR = "../../datasets/raw_phishing_bmp/"
BMP_BASELINE_PATH = "../../datasets/raw_phishing_bmp.json"

# Handle trucated image too
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_bmp_from_png():

    p = pathlib.Path(DEFAULT_TARGET_DIR).glob('**/*')
    files = [x for x in p if x.is_file()]

    print(f"files list : {files}")

    # Convert pictures
    for file in files :
        name_img_source = pathlib.Path(DEFAULT_TARGET_DIR) / file.name
        name_img_target = pathlib.Path(BMP_TARGET_DIR) / pathlib.Path(file.name).with_suffix('.bmp')

        print(name_img_source, name_img_target)

        img = Image.open(name_img_source)
        img.save(name_img_target)

    # Convert baseline file
    json_handler = json_class.JSON_VISUALISATION()
    png_json = json_handler.import_json(pathlib.Path(DEFAULT_BASELINE_PATH)) # PNG one
    bmp_json = json_handler.replace_type(png_json, ".bmp")
    json_handler.json_to_export = bmp_json
    json_handler.json_export(str(BMP_BASELINE_PATH))

if __name__ == '__main__':
    create_bmp_from_png()