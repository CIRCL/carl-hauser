# -*- coding: utf-8 -*-

from .context import *

import unittest

class test_template(unittest.TestCase):
    """Basic test cases."""

    def setUp(self):
        logging = logging.getLogger()
        self.conf = configuration.Default_configuration()
        self.test_file_path = pathlib.Path.cwd() / pathlib.Path("tests/test_files")

    def test_absolute_truth_and_meaning(self):
        assert True


if __name__ == '__main__':
    unittest.main()