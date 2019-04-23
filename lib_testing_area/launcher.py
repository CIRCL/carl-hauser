# ==================== ------------------------ ====================
#                       Configuration launcher
# STD imports
import logging
import pathlib
import json
import traceback

# Own imports
import utility_lib.filesystem_lib as filesystem_lib
import utility_lib.graph_lib as graph_lib
import configuration
import ImageHash.imagehash_test as image_hash
import TLSH.tlsh_test as tlsh
import OpenCV.opencv as opencv
import OpenCV.bow as bow
import Void_baseline.void_baseline as void_baseline

TO_ROUND = 5

class Configuration_launcher():
    def __init__(self,
                 source_pictures_dir: pathlib.Path,
                 output_folder: pathlib.Path,
                 ground_truth_json: pathlib.Path,
                 img_type: configuration.SUPPORTED_IMAGE_TYPE,
                 overwrite_folder : bool):

        # /!\ Logging doesn't work in IDE, but works in terminal /!\

        # logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger()  # See : https://stackoverflow.com/questions/50714316/how-to-use-logging-getlogger-name-in-multiple-modules

        self.source_pictures_dir = source_pictures_dir
        self.output_folder = output_folder
        self.ground_truth_json = ground_truth_json
        self.img_type = img_type

        self.logger.warning("Creation of filesystem handler : deletion of 0-sized pictures in source folder.")
        tmp_conf = configuration.Default_configuration()
        tmp_conf.IMG_TYPE = img_type
        self.filesystem_handler = filesystem_lib.File_System(conf=tmp_conf)
        self.filesystem_handler.clean_folder(self.source_pictures_dir)

        self.overwrite_folder = overwrite_folder

    def auto_launch(self):
        self.logger.info("==== ----- LAUNCHING AUTO CONF LAUNCHER ---- ==== ")
        # self.auto_launch_image_hash()
        # self.auto_launch_tlsh()
        self.auto_launch_orb()
        # self.auto_launch_orb_BOW()
        # self.auto_launch_void()

    def auto_launch_image_hash(self):
        self.logger.info("==== ----- LAUNCHING IMAGE HASH ALGOS ---- ==== ")

        # Create conf
        curr_configuration = configuration.Default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST = [] # No saving
        curr_configuration.OUTPUT_DIR = self.output_folder

        list_to_execute = [configuration.ALGO_TYPE.A_HASH,
                           configuration.ALGO_TYPE.P_HASH,
                           configuration.ALGO_TYPE.P_HASH_SIMPLE,
                           configuration.ALGO_TYPE.D_HASH,
                           configuration.ALGO_TYPE.D_HASH_VERTICAL,
                           configuration.ALGO_TYPE.W_HASH]

        # Launch
        for type in list_to_execute:
            curr_configuration.ALGO = type
            curr_configuration.OUTPUT_DIR = self.output_folder / image_hash.Image_hash_execution_handler.conf_to_string(curr_configuration)

            # Jump to next configuration if we are not overwriting current results
            if not self.overwrite_folder and curr_configuration.OUTPUT_DIR.exists() :
                self.logger.warning(f"Configuration skipped, no overwrite and already exists : {curr_configuration.OUTPUT_DIR}")
                continue

            try:
                self.logger.info(f"Current configuration : {curr_configuration.__dict__} ")
                eh = image_hash.Image_hash_execution_handler(conf=curr_configuration)
                eh.do_full_test()
            except Exception as e:
                self.logger.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")
                traceback.print_tb(e.__traceback__)

    def auto_launch_tlsh(self):
        self.logger.info("==== ----- LAUNCHING TLSH algos ---- ==== ")

        # Create conf
        curr_configuration = configuration.Default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST =  [] # No saving
        curr_configuration.OUTPUT_DIR = self.output_folder

        # Launch
        list_to_execute = [configuration.ALGO_TYPE.TLSH,
                           configuration.ALGO_TYPE.TLSH_NO_LENGTH]

        # Launch
        for type in list_to_execute:
            curr_configuration.ALGO = type
            curr_configuration.OUTPUT_DIR = self.output_folder / tlsh.TLSH_execution_handler.conf_to_string(curr_configuration)

            # Jump to next configuration if we are not overwriting current results
            if not self.overwrite_folder and curr_configuration.OUTPUT_DIR.exists() :
                self.logger.warning(f"Configuration skipped, no overwrite and already exists : {curr_configuration.OUTPUT_DIR}")
                continue

            try:
                self.logger.info(f"Current configuration : {curr_configuration.__dict__} ")
                eh = tlsh.TLSH_execution_handler(conf=curr_configuration)
                eh.do_full_test()
            except Exception as e:
                self.logger.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")
                traceback.print_tb(e.__traceback__)

    def auto_launch_orb(self):
        self.logger.info("==== ----- LAUNCHING ORB algos ---- ==== ")

        # Create conf
        curr_configuration = configuration.ORB_default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST = [] # No saving
        curr_configuration.OUTPUT_DIR = self.output_folder

        curr_configuration.ALGO = configuration.ALGO_TYPE.ORB
        curr_configuration.ORB_KEYPOINTS_NB = 500

        for match in configuration.MATCH_TYPE:
            for datastruct in configuration.DATASTRUCT_TYPE:
                for filter in configuration.FILTER_TYPE:
                    for distance in configuration.DISTANCE_TYPE:
                        for crosscheck in [configuration.CROSSCHECK.DISABLED, configuration.CROSSCHECK.ENABLED]:

                            # Check configuration
                            if filter == configuration.FILTER_TYPE.FAR_THREESHOLD and match == configuration.MATCH_TYPE.STD :
                                continue
                            # if filter == configuration.FILTER_TYPE.FAR_THREESHOLD and match == configuration.MATCH_TYPE.STD :
                            #     continue
                            # cat ./raw_phishing_output.overview | grep -v "LEAN_MEAN" | grep -v "FLANN_KDTREE" | grep "FAR_THREESHOLD_STD"

                            curr_configuration.MATCH = match
                            curr_configuration.DATASTRUCT = datastruct
                            curr_configuration.FILTER = filter
                            curr_configuration.DISTANCE = distance
                            curr_configuration.CROSSCHECK = crosscheck

                            curr_configuration.OUTPUT_DIR = self.output_folder / opencv.OpenCV_execution_handler.conf_to_string(curr_configuration)

                            # Jump to next configuration if we are not overwriting current results
                            if not self.overwrite_folder and curr_configuration.OUTPUT_DIR.exists():
                                self.logger.warning(f"Configuration skipped, no overwrite and already exists : {curr_configuration.OUTPUT_DIR}")
                                continue

                            try:
                                self.logger.info(f"Current configuration : {curr_configuration.__dict__} ")
                                eh = opencv.OpenCV_execution_handler(conf=curr_configuration)
                                eh.do_full_test()
                            except Exception as e:
                                self.logger.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")
                                traceback.print_tb(e.__traceback__)

    def auto_launch_orb_BOW(self):
        self.logger.info("==== ----- LAUNCHING ORB algos ---- ==== ")

        # Create conf
        curr_configuration = configuration.BoW_ORB_default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST =  [configuration.PICTURE_SAVE_MODE.RANSAC_MATRIX] # No saving
        curr_configuration.OUTPUT_DIR = self.output_folder

        curr_configuration.ALGO = configuration.ALGO_TYPE.ORB
        curr_configuration.ORB_KEYPOINTS_NB = 500


        large_size_set = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        small_size_set = [100, 1000, 10000, 100000, 1000000]

        for size in small_size_set:
            for hist_cmp in configuration.BOW_CMP_HIST:

                curr_configuration.BOW_SIZE = size
                curr_configuration.BOW_CMP_HIST = hist_cmp

                curr_configuration.OUTPUT_DIR = self.output_folder / opencv.OpenCV_execution_handler.conf_to_string(curr_configuration)

                # Jump to next configuration if we are not overwriting current results
                if not self.overwrite_folder and curr_configuration.OUTPUT_DIR.exists():
                    self.logger.warning(f"Configuration skipped, no overwrite and already exists : {curr_configuration.OUTPUT_DIR}")
                    continue

                try:
                    self.logger.info(f"Current configuration : {curr_configuration.__dict__} ")
                    eh = bow.BoW_execution_handler(conf=curr_configuration)
                    eh.do_full_test()
                except Exception as e:
                    self.logger.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")
                    traceback.print_tb(e.__traceback__)


    def auto_launch_void(self):
        self.logger.info("==== ----- LAUNCHING Void baseline ---- ==== ")

        # Create conf
        curr_configuration = configuration.Default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST =  [] # No saving
        curr_configuration.OUTPUT_DIR = self.output_folder / "void_baseline"

        self.logger.info(f"Current configuration : {curr_configuration.__dict__} ")
        eh = void_baseline.Void_baseline(conf=curr_configuration)
        eh.do_full_test()


    @staticmethod
    def create_tldr(folder: pathlib.Path, output_file: pathlib.Path):
        f = open(str(output_file.resolve()), "w+")  # Append and create if does not exist

        global_list = []
        for x in folder.resolve().iterdir():

            global_txt = ""

            if x.is_dir():
                global_txt += (x.name).ljust(95, " ") + "\t"
                stat_file = x / "stats.txt"
                if stat_file.exists():

                    data = filesystem_lib.File_System.load_json(stat_file)
                    LEN = 34
                    global_txt += ("TRUE_POSITIVE = " + str(data["TRUE_POSITIVE_RATE"])).ljust(LEN, " ") + " \t"
                    global_txt += ("PRE_COMPUTING = " + str(data["TIME_PER_PICTURE_PRE_COMPUTING"])).ljust(LEN, " ") + " \t"
                    global_txt += ("MATCHING = " + str(data["TIME_PER_PICTURE_MATCHING"])).ljust(LEN, " ")

                    if hasattr(data, "COMPUTED_THREESHOLD") : # Backwards compatibility with previously generated stats
                        global_txt += ("THREESHOLD DIST = " + str(data["COMPUTED_THREESHOLD"])).ljust(LEN, " ")
                    if hasattr(data, "TRUE_POSITIVE_RATE_THREESHOLD") : # Backwards compatibility with previously generated stats
                        global_txt += ("TRUE_POSITIVE_W_T = " + str(data["TRUE_POSITIVE_RATE_THREESHOLD"])).ljust(LEN, " ")

                    global_list.append([global_txt, data["TRUE_POSITIVE_RATE"]])

                else:
                    global_txt += "NO RESULT / ERROR"

                    global_list.append([global_txt, -1])

        global_list = sorted(global_list, key=lambda l: l[1], reverse=True)

        for x in global_list:
            f.write(x[0] + "\r\n")
        f.close()

        logger = logging.getLogger(__name__)
        logger.info("Overview written")


    ##     Conf & nobs & min time &  max time & mean & variance & skewness & kurtosis & True Positive\\ \hline
    ## ORB \\ LEN MAX - KNN 2 \\Crosscheck : False \\FLANN LSH \\FAR THREESHOLD & 190 & 0.26489s & 1.57223s & 1.11384s & 0.04294s & -0.97579s & 1.09073 & 0.63158 \\ \hline

    @staticmethod
    def create_latex_tldr(folder: pathlib.Path, output_file: pathlib.Path):
        f = open(str(output_file.resolve()), "w+")  # Append and create if does not exist
        f.write("NAME & TRUE POSITIVE & PRE COMPUTING (sec) & MATCHING (sec)" + "\\\\ \hline \r\n")

        global_list = []
        for x in folder.resolve().iterdir():

            global_txt = ""

            if x.is_dir():
                global_txt += (x.name).replace("_", " ") + " & "
                stat_file = x / "stats.txt"
                if stat_file.exists():

                    data = filesystem_lib.File_System.load_json(stat_file)
                    global_txt += str(round(data["TRUE_POSITIVE_RATE"], TO_ROUND)) + " & "
                    global_txt += str(round(data["TIME_PER_PICTURE_PRE_COMPUTING"], TO_ROUND)) + " & "
                    global_txt += str(round(data["TIME_PER_PICTURE_MATCHING"], TO_ROUND)) + "\\\\ \hline "

                    global_list.append([global_txt, data["TRUE_POSITIVE_RATE"]])

                # else:
                #     global_txt += "NO RESULT / ERROR"
                #     continue # Jump without adding it

        global_list = sorted(global_list, key=lambda l: l[1], reverse=True)

        for x in global_list:
            f.write(x[0] + "\r\n")
        f.close()

        logger = logging.getLogger(__name__)
        logger.info("Latex overview written")


    @staticmethod
    def create_and_export_inclusion_matrix(folder: pathlib.Path, output_file: pathlib.Path):
        global_result = graph_lib.Graph_handler.create_inclusion_matrix(folder=folder)
        graph_lib.Graph_handler.save_matrix_to_json(global_result, output_file.with_suffix(".json"))

        ordo, absi, values = graph_lib.Graph_handler.inclusion_matrix_to_triple_array(global_result)

        graph = graph_lib.Graph_handler()
        graph.set_values(ordo, absi, values)

        graph.save_matrix(output_file.with_suffix(".pdf"))


    @staticmethod
    def create_and_export_pair_matrix(input_folder: pathlib.Path, ground_truth_json: pathlib.Path, output_file: pathlib.Path):
        # Generate pairs
        global_result = graph_lib.Graph_handler.create_pair_matrix(folder=input_folder, ground_truth_json=ground_truth_json)

        # Save the pair results
        graph_lib.Graph_handler.save_matrix_to_json(global_result, output_file.with_suffix(".json"))

        # Build the matrix
        ordo, absi, values = graph_lib.Graph_handler.inclusion_matrix_to_triple_array(global_result)

        graph = graph_lib.Graph_handler()
        graph.set_values(ordo, absi, values)

        graph.save_matrix(output_file.with_suffix(".pdf"))


    @staticmethod
    def create_paired_results(input_folder: pathlib.Path, target_pair_folder: pathlib.Path, ground_truth_json: pathlib.Path):
        # Generate pairs
        graph_lib.Graph_handler.generate_merged_pairs(input_folder=input_folder, target_pair_folder=target_pair_folder)

        # Evaluate each graphe
        graph_lib.Graph_handler.evaluate_graphs(target_pair_folder=target_pair_folder, ground_truth_json=ground_truth_json)


# For profiling :
# import cProfile

if __name__ == '__main__':

    logger = logging.getLogger()  # See : https://stackoverflow.com/questions/50714316/how-to-use-logging-getlogger-name-in-multiple-modules
    logger.setLevel(logging.INFO)
    # create console handler with a higher log level
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    # create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - + %(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handler to the logger
    logger.addHandler(handler)

    base_path = [["../datasets/raw_phishing", configuration.SUPPORTED_IMAGE_TYPE.PNG],
                 # ["../datasets/raw_phishing_bmp", configuration.SUPPORTED_IMAGE_TYPE.BMP],
                 # ["../datasets/raw_phishing_COLORED", configuration.SUPPORTED_IMAGE_TYPE.PNG],
                 # ["../datasets/raw_phishing_Tesseract", configuration.SUPPORTED_IMAGE_TYPE.PNG],
                 # ["../datasets/raw_phishing_EDGES_CANNY", configuration.SUPPORTED_IMAGE_TYPE.PNG],
                 # ["../datasets/raw_phishing_EDGES_DEEP", configuration.SUPPORTED_IMAGE_TYPE.PNG]
                 ]

    for curr_base_path, img_type in base_path:

        # =============================
        # Source folder with raw pictures
        source_pictures_dir = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "/")
        # Source ground truth json
        ground_truth_json = pathlib.Path.cwd() / pathlib.Path(curr_base_path + ".json")
        # Output folder for statistics of executions folders
        output_folder = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output/")
        if not output_folder.resolve().exists() : output_folder.resolve().mkdir()

        # Outut folder for statics of executions for paired results
        paired_output_folder = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output_paired/")
        if not paired_output_folder.resolve().exists() : paired_output_folder.resolve().mkdir()

        # Output overview files
        output_overview_file = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output.overview")
        output_latex_overview_file = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output.latex.overview")
        output_overview_paired_file = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output_paired.overview")

        # Output matrix files
        output_similarity_matrix = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output.matrix")
        output_paired_matrix = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output_paired.matrix")

        # =============================
        # img_type = configuration.SUPPORTED_IMAGE_TYPE.PNG
        config_launcher = Configuration_launcher(source_pictures_dir=source_pictures_dir.resolve(),
                                                 output_folder=output_folder.resolve(),
                                                 ground_truth_json=ground_truth_json.resolve(),
                                                 img_type=img_type,
                                                 overwrite_folder=False)
        # For profiling : cProfile.run("
        config_launcher.auto_launch()
        # ")

        # Create overview for simple results
        Configuration_launcher.create_tldr(folder=output_folder, output_file=output_overview_file)
        Configuration_launcher.create_latex_tldr(folder=output_folder, output_file=output_latex_overview_file)

        # Create overview for paired results
        # Configuration_launcher.create_paired_results(input_folder=output_folder, target_pair_folder=paired_output_folder, ground_truth_json=ground_truth_json)
        # Configuration_launcher.create_tldr(folder=paired_output_folder, output_file=output_overview_paired_file)

        # Create matrixes
        # Configuration_launcher.create_and_export_inclusion_matrix(folder=output_folder, output_file=output_similarity_matrix)
        # Configuration_launcher.create_and_export_pair_matrix(input_folder=output_folder, ground_truth_json=ground_truth_json, output_file=output_paired_matrix)
