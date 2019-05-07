import logging
import pathlib
import argparse

import pathlib
import json
import traceback
import pprint

# Own imports
import utility_lib.filesystem_lib as filesystem_lib
import utility_lib.graph_lib as graph_lib
import configuration
import ImageHash.imagehash_test as image_hash
import TLSH.tlsh_test as tlsh
import OpenCV.opencv as opencv
import OpenCV.bow as bow
import Void_baseline.void_baseline as void_baseline

class Configuration_launcher():
    def __init__(self,
                 source_pictures_dir: pathlib.Path,
                 output_folder: pathlib.Path,
                 ground_truth_json: pathlib.Path,
                 img_type: configuration.SUPPORTED_IMAGE_TYPE,
                 overwrite_folder : bool,
                 args):

        # /!\ Logging doesn't work in IDE, but works in terminal /!\

        # logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger()  # See : https://stackoverflow.com/questions/50714316/how-to-use-logging-getlogger-name-in-multiple-modules

        self.source_pictures_dir = source_pictures_dir
        self.output_folder = output_folder
        self.ground_truth_json = ground_truth_json
        self.img_type = img_type

        tmp_conf = configuration.Default_configuration()
        tmp_conf.IMG_TYPE = img_type

        self.logger.warning("Creation of filesystem handler : deletion of 0-sized pictures in source folder.")
        self.filesystem_handler = filesystem_lib.File_System(conf=tmp_conf)
        self.filesystem_handler.clean_folder(self.source_pictures_dir)

        self.overwrite_folder = overwrite_folder

        self.args = args

    def add_logfile(self, curr_configuration):
        if not curr_configuration.OUTPUT_DIR.exists() : curr_configuration.OUTPUT_DIR.mkdir()
        logger = logging.getLogger()  # See : https://stackoverflow.com/questions/50714316/how-to-use-logging-getlogger-name-in-multiple-modules
        tmp_log_file_handler = logging.FileHandler(str(curr_configuration.OUTPUT_DIR / pathlib.Path('execution.log')),
                                                   'w')
        tmp_log_file_handler.setLevel(logging.INFO)
        tmp_log_file_handler.setFormatter(configuration.FORMATTER)
        logger.addHandler(tmp_log_file_handler)

        return tmp_log_file_handler

    def rem_logfile(self, loghandler):
        logger = logging.getLogger()  # See : https://stackoverflow.com/questions/50714316/how-to-use-logging-getlogger-name-in-multiple-modules

        logger.handlers = [
            h for h in logger.handlers if not h == loghandler]

    def launch_exec_handler(self, exec_handler, curr_configuration):
        tmp_log_handler = self.add_logfile(curr_configuration)

        try:
            self.logger.info(f"Current configuration : \n {pprint.pformat(curr_configuration.__dict__)}")

            eh = exec_handler(conf=curr_configuration)
            eh.do_full_test()
        except Exception as e:
            self.logger.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")
            self.logger.error(traceback.print_tb(e.__traceback__))
        finally:
            self.rem_logfile(tmp_log_handler)

    def skip_if_already_computed(self, curr_configuration):
        # Jump to next configuration if we are not overwriting current results
        if not self.overwrite_folder and curr_configuration.OUTPUT_DIR.exists():
            self.logger.warning(f"Configuration skipped, no overwrite and already exists : {curr_configuration.OUTPUT_DIR} \n")
            return True
        elif self.overwrite_folder :
            self.logger.info(f"Configuration overwriten. Name generation : {curr_configuration.OUTPUT_DIR}")
            return False
        else :
            self.logger.info(f"Configuration absent. Name generation : {curr_configuration.OUTPUT_DIR}")
            return False

    def auto_launch(self):
        self.logger.info("==== ----- LAUNCHING AUTO CONF LAUNCHER ---- ==== ")
        if self.args.imagehash :     self.auto_launch_image_hash()
        if self.args.tlsh :          self.auto_launch_tlsh()
        if self.args.orb_normal :    self.auto_launch_orb()
        if self.args.orb_bow :       self.auto_launch_orb_BOW()
        if self.args.void :          self.auto_launch_void()


    def auto_launch_image_hash(self):
        self.logger.info("==== ----- LAUNCHING IMAGE HASH ALGOS ---- ==== ")

        # Create conf
        curr_configuration = configuration.Default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        if self.args.save_pictures :
            curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST = [configuration.PICTURE_SAVE_MODE.TOP3]
        else :
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
            if self.skip_if_already_computed(curr_configuration) : continue

            # Launch configuration
            self.launch_exec_handler(image_hash.Image_hash_execution_handler, curr_configuration)

    def auto_launch_tlsh(self):
        self.logger.info("==== ----- LAUNCHING TLSH algos ---- ==== ")

        # Create conf
        curr_configuration = configuration.Default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        if self.args.save_pictures :
            curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST = [configuration.PICTURE_SAVE_MODE.TOP3]
        else :
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
            if self.skip_if_already_computed(curr_configuration) : continue

            # Launch configuration
            self.launch_exec_handler(tlsh.TLSH_execution_handler, curr_configuration)

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

        if self.args.save_pictures :
            saving_list = [configuration.PICTURE_SAVE_MODE.TOP3,
                           configuration.PICTURE_SAVE_MODE.FEATURE_MATCHES_TOP3,
                           configuration.PICTURE_SAVE_MODE.RANSAC_MATRIX]
        else :
            saving_list = [] # No saving

        for match in configuration.MATCH_TYPE:
            for datastruct in configuration.DATASTRUCT_TYPE:
                for filter in configuration.FILTER_TYPE:
                    for distance in configuration.DISTANCE_TYPE:
                        for crosscheck in [configuration.CROSSCHECK.DISABLED, configuration.CROSSCHECK.ENABLED]:
                            for postfilter in configuration.POST_FILTER:

                                # Check configuration
                                if filter == configuration.FILTER_TYPE.FAR_THREESHOLD and match == configuration.MATCH_TYPE.STD :
                                    self.logger.warning("Detected FAR THRESHOLD + STD MATCH : configuration aborted.\n")
                                    continue

                                if filter != configuration.FILTER_TYPE.RANSAC and postfilter == configuration.POST_FILTER.MATRIX_CHECK:
                                    self.logger.warning("Detected not RANSAC + with RANSAC MATRIX_CHECK FILTER : configuration aborted.\n")
                                    continue

                                if filter != configuration.FILTER_TYPE.RANSAC and configuration.PICTURE_SAVE_MODE.RANSAC_MATRIX in saving_list:
                                    tmp_saving_list = saving_list.copy()
                                    tmp_saving_list.remove(configuration.PICTURE_SAVE_MODE.RANSAC_MATRIX)
                                    self.logger.warning("Detected RANSAC matches saving mode without RANSAC filter. Removing this instruction for this execution configuration.\n")
                                else :
                                    tmp_saving_list = saving_list

                                if filter == configuration.FILTER_TYPE.RATIO_CORRECT and match == configuration.MATCH_TYPE.STD:
                                    self.logger.warning("Detected RATIO CORRECT + STD MATCH : configuration aborted.\n")
                                    continue

                                # if filter == configuration.FILTER_TYPE.FAR_THREESHOLD and match == configuration.MATCH_TYPE.STD :
                                #     continue
                                # cat ./raw_phishing_output.overview | grep -v "LEAN_MEAN" | grep -v "FLANN_KDTREE" | grep "FAR_THREESHOLD_STD"

                                curr_configuration.MATCH = match
                                curr_configuration.DATASTRUCT = datastruct
                                curr_configuration.FILTER = filter
                                curr_configuration.DISTANCE = distance
                                curr_configuration.CROSSCHECK = crosscheck
                                curr_configuration.POST_FILTER_CHOSEN = postfilter
                                curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST = tmp_saving_list

                                curr_configuration.OUTPUT_DIR = self.output_folder / opencv.OpenCV_execution_handler.conf_to_string(curr_configuration)

                                # Jump to next configuration if we are not overwriting current results
                                if self.skip_if_already_computed(curr_configuration): continue

                                # Launch configuration
                                self.launch_exec_handler(opencv.OpenCV_execution_handler, curr_configuration)


    def auto_launch_orb_BOW(self):
        self.logger.info("==== ----- LAUNCHING ORB algos ---- ==== ")

        # Create conf
        curr_configuration = configuration.BoW_ORB_default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type

        if self.args.save_pictures :
            curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST = [configuration.PICTURE_SAVE_MODE.TOP3,
                                                                configuration.PICTURE_SAVE_MODE.FEATURE_MATCHES_TOP3]
        else :
            curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST = [] # No saving

        curr_configuration.OUTPUT_DIR = self.output_folder

        curr_configuration.ALGO = configuration.ALGO_TYPE.ORB
        curr_configuration.ORB_KEYPOINTS_NB = 500

        # large_size_set = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        small_size_set = [100, 1000, 10000, 100000, 1000000]

        for size in small_size_set:
            for hist_cmp in configuration.BOW_CMP_HIST:

                curr_configuration.BOW_SIZE = size
                curr_configuration.BOW_CMP_HIST = hist_cmp

                curr_configuration.OUTPUT_DIR = self.output_folder / opencv.OpenCV_execution_handler.conf_to_string(curr_configuration)

                # Jump to next configuration if we are not overwriting current results
                if self.skip_if_already_computed(curr_configuration): continue

                # Launch configuration
                self.launch_exec_handler(bow.BoW_execution_handler, curr_configuration)

    def auto_launch_void(self):
        self.logger.info("==== ----- LAUNCHING Void baseline ---- ==== ")

        # Create conf
        curr_configuration = configuration.Default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST =  [] # No saving
        curr_configuration.OUTPUT_DIR = self.output_folder / "void_baseline"

        # Launch configuration
        self.launch_exec_handler(void_baseline.Void_baseline, curr_configuration)

    @staticmethod
    def create_tldr(folder: pathlib.Path, output_file: pathlib.Path):
        logger = logging.getLogger()

        f = open(str(output_file.resolve()), "w+")  # Append and create if does not exist

        global_list = []
        for x in folder.resolve().iterdir():

            global_txt = ""

            if x.is_dir():
                global_txt += (x.name).ljust(95, " ") + "\t"
                stat_file = x / "stats.txt"
                if stat_file.exists():

                    try :
                        data = filesystem_lib.File_System.load_json(stat_file)
                    except Exception as e:
                        logger.error(f"Impossible to load {stat_file}")

                        global_txt += "FILE READING ERROR (JSON LOAD)"

                        global_list.append([global_txt, -1])

                    else :
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
        logger = logging.getLogger()

        f = open(str(output_file.resolve()), "w+")  # Append and create if does not exist
        f.write("NAME & TRUE POSITIVE & PRE COMPUTING (sec) & MATCHING (sec)" + "\\\\ \hline \r\n")

        global_list = []
        for x in folder.resolve().iterdir():

            global_txt = ""

            if x.is_dir():
                global_txt += (x.name).replace("_", " ") + " & "
                stat_file = x / "stats.txt"
                if stat_file.exists():

                    try :
                        data = filesystem_lib.File_System.load_json(stat_file)
                    except Exception as e:
                        logger.error(f"Impossible to load {stat_file}")
                    else :
                        global_txt += str(round(data["TRUE_POSITIVE_RATE"], configuration.TO_ROUND)) + " & "
                        global_txt += str(round(data["TIME_PER_PICTURE_PRE_COMPUTING"], configuration.TO_ROUND)) + " & "
                        global_txt += str(round(data["TIME_PER_PICTURE_MATCHING"], configuration.TO_ROUND)) + "\\\\ \hline "

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
