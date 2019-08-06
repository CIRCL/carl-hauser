# -*- coding: utf-8 -*-

from .context import *

import unittest


class test_template(unittest.TestCase):
    """Basic test cases."""

    def setUp(self):
        self.logger = logging.getLogger()
        self.conf = configuration.Default_configuration()
        self.test_file_path = pathlib.Path.cwd() / pathlib.Path("tests/test_files")

        self.curr_configuration = configuration.ORB_default_configuration()
        self.curr_configuration.SOURCE_DIR = self.test_file_path / "MINI_DATASET"
        self.curr_configuration.GROUND_TRUTH_PATH = self.test_file_path / "MINI_DATASET.json"
        self.curr_configuration.IMG_TYPE = configuration.SUPPORTED_IMAGE_TYPE.PNG
        self.curr_configuration.SAVE_PICTURE = False
        self.curr_configuration.OUTPUT_DIR = self.test_file_path / "OpenCV"

        # Create conf
        self.curr_configuration.ALGO = configuration.ALGO_TYPE.ORB
        self.curr_configuration.ORB_KEYPOINTS_NB = 500

    def test_absolute_truth_and_meaning(self):
        self.assertTrue(True)

    def test_BASIC(self):
        self.curr_configuration.OUTPUT_DIR = self.curr_configuration.OUTPUT_DIR / "STD"

        self.curr_configuration.MATCH = configuration.MATCH_TYPE.STD
        self.curr_configuration.DATASTRUCT = configuration.DATASTRUCT_TYPE.BRUTE_FORCE
        self.curr_configuration.FILTER = configuration.FILTER_TYPE.NO_FILTER
        self.curr_configuration.DISTANCE = configuration.DISTANCE_TYPE.LEN_MAX
        self.curr_configuration.CROSSCHECK = configuration.CROSSCHECK.AUTO

        try :
            eh = opencv.OpenCV_execution_handler(conf=self.curr_configuration)
            eh.do_full_test()
            self.assertTrue(True)
        except Exception as e:
            self.logger.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")
            traceback.print_tb(e.__traceback__)
            self.assertTrue(False)

    def test_RANSAC(self):
        self.curr_configuration.OUTPUT_DIR = self.curr_configuration.OUTPUT_DIR / "RANSAC"

        self.curr_configuration.MATCH = configuration.MATCH_TYPE.STD
        self.curr_configuration.DATASTRUCT = configuration.DATASTRUCT_TYPE.BRUTE_FORCE
        self.curr_configuration.FILTER = configuration.FILTER_TYPE.RANSAC
        self.curr_configuration.DISTANCE = configuration.DISTANCE_TYPE.LEN_MAX
        self.curr_configuration.CROSSCHECK = configuration.CROSSCHECK.AUTO

        try :
            eh = opencv.OpenCV_execution_handler(conf=self.curr_configuration)
            eh.do_full_test()
            self.assertTrue(True)
        except Exception as e:
            self.logger.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")
            traceback.print_tb(e.__traceback__)
            self.assertTrue(False)

if __name__ == '__main__':
    unittest.main()
