# ==================== ------------------------ ====================
#                       Configuration launcher
# STD imports
import logging
import pathlib

# Own imports
import utility_lib.filesystem_lib as filesystem_lib
import configuration
import ImageHash.imagehash_test as image_hash
import TLSH.tlsh_test as tlsh
import OpenCV.opencv as opencv


class Configuration_launcher():
    def __init__(self,
                 source_pictures_dir: pathlib.Path,
                 output_folder: pathlib.Path,
                 ground_truth_json: pathlib.Path,
                 img_type: configuration.SUPPORTED_IMAGE_TYPE):
        self.logger = logging.getLogger(__name__)
        self.source_pictures_dir = source_pictures_dir
        self.output_folder = output_folder
        self.ground_truth_json = ground_truth_json
        self.img_type = img_type

        self.logger.warning("Creation of filesystem handler : deletion of 0-sized pictures in source folder.")
        tmp_conf = configuration.Default_configuration()
        tmp_conf.IMG_TYPE = img_type
        self.filesystem_handler = filesystem_lib.File_System(conf=tmp_conf)
        self.filesystem_handler.clean_folder(self.source_pictures_dir)

    def auto_launch(self):
        self.logger.info("==== ----- LAUNCHING AUTO CONF LAUNCHER ---- ==== ")
        self.auto_launch_image_hash()
        self.auto_launch_tlsh()
        self.auto_launch_orb()

    def auto_launch_image_hash(self):
        self.logger.info("==== ----- LAUNCHING IMAGE HASH ALGOS ---- ==== ")

        # Create conf
        curr_configuration = configuration.Default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        curr_configuration.SAVE_PICTURE = False
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
            try:
                eh = image_hash.Image_hash_execution_handler(conf=curr_configuration)
                eh.do_full_test()
            except Exception as e:
                logging.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")

    def auto_launch_tlsh(self):
        self.logger.info("==== ----- LAUNCHING TLSH algos ---- ==== ")

        # Create conf
        curr_configuration = configuration.Default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        curr_configuration.SAVE_PICTURE = False
        curr_configuration.OUTPUT_DIR = self.output_folder

        # Launch
        list_to_execute = [configuration.ALGO_TYPE.TLSH,
                           configuration.ALGO_TYPE.TLSH_NO_LENGTH]

        # Launch
        for type in list_to_execute:
            curr_configuration.ALGO = type
            try:
                eh = tlsh.TLSH_execution_handler(conf=curr_configuration)
                eh.do_full_test()
            except Exception as e:
                logging.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")

    def auto_launch_orb(self):
        self.logger.info("==== ----- LAUNCHING ORB algos ---- ==== ")

        # Create conf
        curr_configuration = configuration.ORB_default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        curr_configuration.SAVE_PICTURE = False
        curr_configuration.OUTPUT_DIR = self.output_folder

        curr_configuration.ALGO = configuration.ALGO_TYPE.ORB
        curr_configuration.ORB_KEYPOINTS_NB = 500

        for match in configuration.MATCH_TYPE:
            for datastruct in configuration.DATASTRUCT_TYPE:
                for filter in configuration.FILTER_TYPE:
                    for distance in configuration.DISTANCE_TYPE:
                        for crosscheck in [configuration.CROSSCHECK.DISABLED, configuration.CROSSCHECK.ENABLED]:

                            curr_configuration.MATCH = match
                            curr_configuration.DATASTRUCT = datastruct
                            curr_configuration.FILTER = filter
                            curr_configuration.DISTANCE = distance
                            curr_configuration.CROSSCHECK = crosscheck

                            try:
                                eh = opencv.OpenCV_execution_handler(conf=curr_configuration)
                                eh.do_full_test()
                            except Exception as e:
                                logging.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")


if __name__ == '__main__':
    source_pictures_dir = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing/")
    output_folder = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_output/")
    ground_truth_json = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing.json")
    img_type = configuration.SUPPORTED_IMAGE_TYPE.PNG

    config_launcher = Configuration_launcher(source_pictures_dir=source_pictures_dir.resolve(),
                                             output_folder=output_folder.resolve(),
                                             ground_truth_json=ground_truth_json.resolve(),
                                             img_type=img_type)
    config_launcher.auto_launch()

    target_dir = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_bmp/")
    output_folder = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_bmp_output/")
    ground_truth_json = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_bmp.json")
    img_type = configuration.SUPPORTED_IMAGE_TYPE.BMP

    config_launcher = Configuration_launcher(source_pictures_dir=source_pictures_dir.resolve(),
                                             output_folder=output_folder.resolve(),
                                             ground_truth_json=ground_truth_json.resolve(),
                                             img_type=img_type)
    config_launcher.auto_launch()
