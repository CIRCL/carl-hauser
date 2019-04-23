# -*- coding: utf-8 -*-

from .context import *

import unittest


class test_template(unittest.TestCase):
    """Basic test cases."""

    def setUp(self):
        self.logger = logging.getLogger()

        formatter = logging.Formatter('%(asctime)s - + %(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')

        self.conf = configuration.Default_configuration()
        self.test_file_path = pathlib.Path.cwd() / pathlib.Path("tests/test_files/utility/execution_handler")

        ###
        self.source_pictures_dir = self.test_file_path / "image_folder"
        self.output_folder = self.test_file_path / "output_folder"
        self.ground_truth_json = self.test_file_path / "ground_truth.json"

        self.tmp_conf = configuration.Default_configuration()
        self.tmp_conf.IMG_TYPE = configuration.SUPPORTED_IMAGE_TYPE.PNG
        self.filesystem_handler = filesystem_lib.File_System(conf=self.tmp_conf)

    def test_absolute_truth_and_meaning(self):
        assert True

    def test_do_pair_orb_full(self):
        # Create conf
        curr_configuration = configuration.ORB_default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.OUTPUT_DIR = self.output_folder / "output_folder_orb_full"
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.tmp_conf.IMG_TYPE
        curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST = [configuration.PICTURE_SAVE_MODE.TOP3,
                                                            configuration.PICTURE_SAVE_MODE.FEATURE_MATCHES_TOP3,
                                                            configuration.PICTURE_SAVE_MODE.RANSAC_MATRIX]  # No saving

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

                            curr_configuration.OUTPUT_DIR = self.output_folder / opencv.OpenCV_execution_handler.conf_to_string(curr_configuration)

                            try:
                                self.logger.info(f"Current configuration : {curr_configuration.__dict__} ")
                                eh = opencv.OpenCV_execution_handler(conf=curr_configuration)
                                eh.do_full_test()
                            except Exception as e:
                                self.logger.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")
                                traceback.print_tb(e.__traceback__)

    def test_do_RANSAC_orb(self):
        # Create conf
        curr_configuration = configuration.ORB_default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.OUTPUT_DIR = self.output_folder / "output_folder_orb_full"
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.tmp_conf.IMG_TYPE
        curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST = [configuration.PICTURE_SAVE_MODE.TOP3,
                                                            configuration.PICTURE_SAVE_MODE.FEATURE_MATCHES_TOP3,
                                                            configuration.PICTURE_SAVE_MODE.RANSAC_MATRIX]  # No saving

        curr_configuration.ALGO = configuration.ALGO_TYPE.ORB
        curr_configuration.ORB_KEYPOINTS_NB = 500

        for match in configuration.MATCH_TYPE:
            for datastruct in configuration.DATASTRUCT_TYPE:
                for distance in configuration.DISTANCE_TYPE:
                    for crosscheck in [configuration.CROSSCHECK.DISABLED, configuration.CROSSCHECK.ENABLED]:

                        curr_configuration.MATCH = match
                        curr_configuration.DATASTRUCT = datastruct
                        curr_configuration.FILTER = configuration.FILTER_TYPE.RANSAC
                        curr_configuration.DISTANCE = distance
                        curr_configuration.CROSSCHECK = crosscheck

                        curr_configuration.OUTPUT_DIR = self.output_folder / opencv.OpenCV_execution_handler.conf_to_string(curr_configuration)

                        try:
                            self.logger.info(f"Current configuration : {curr_configuration.__dict__} ")
                            eh = opencv.OpenCV_execution_handler(conf=curr_configuration)
                            eh.do_full_test()
                        except Exception as e:
                            self.logger.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")
                            traceback.print_tb(e.__traceback__)

    def test_do_pair_orb_pairwise(self):
        pic1_path = self.source_pictures_dir / "1.png"
        pic2_path = self.source_pictures_dir / "2.png"

        # Create conf
        curr_configuration = configuration.ORB_default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.OUTPUT_DIR = self.output_folder / "output_folder_orb_pairwise"
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.tmp_conf.IMG_TYPE
        curr_configuration.SAVE_PICTURE_INSTRUCTION_LIST = []  # No saving

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

                            curr_configuration.OUTPUT_DIR = self.output_folder / opencv.OpenCV_execution_handler.conf_to_string(curr_configuration)

                            try:
                                self.logger.info(f"Current configuration : {curr_configuration.__dict__} ")
                                eh = opencv.OpenCV_execution_handler(conf=curr_configuration)
                                eh.do_pair_test(pic1_path, pic2_path)
                            except Exception as e:
                                self.logger.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")
                                traceback.print_tb(e.__traceback__)


if __name__ == '__main__':
    unittest.main()
