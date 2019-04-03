# -*- coding: utf-8 -*-

from .context import *

import unittest

DEBUG = False


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def setUp(self):
        self.JSON_class = json_class.Json_handler()
        self.test_file_path = pathlib.Path.cwd() / pathlib.Path("tests/test_files")

        if DEBUG:
            print(str(self.test_file_path))

    def test_absolute_truth_and_meaning(self):
        assert True

    def test_import_json(self):
        json_imported = self.JSON_class.import_json(self.test_file_path / 'mini_baseline.json')
        self.assertEqual(json_imported["nodes"][0]["id"], 0)
        self.assertEqual(json_imported["nodes"][1]["image"], 'www.force-xloot.ml.png')

    def test_mapping_json(self):
        json_imported =  self.JSON_class.import_json(self.test_file_path / 'mini_baseline.json')
        self.JSON_class.json_to_export = json_imported
        json_imported = self.JSON_class.import_json(self.test_file_path / 'mini_baseline_to_match.json')

        self.mapping_dict = json_class.create_node_mapping(self.JSON_class.json_to_export, json_imported)

        if DEBUG:
            print("Matching : ")
            print(self.JSON_class.json_to_export)
            print("with : ")
            print(json_imported)
            print("gives : ")
            print(self.mapping_dict)

        self.assertEqual(self.mapping_dict[0], 1)
        self.assertEqual(self.mapping_dict[1], 2)
        self.assertEqual(self.mapping_dict[2], 0)

    def print_debug_are_same_edge_simple(self, edge1, mapping, edge2):
        if DEBUG:
            print("Edge 1 : ")
            print(edge1)
            print("Edge 2 : ")
            print(edge2)
            print("Mapping ")
            print(mapping)

    def test_are_same_edge_simple(self):
        edge1 = {"to"  : 0,"from": 1}
        edge2 = {"to"  : 0,"from": 1}

        mapping = {0:0,1:1}
        self.print_debug_are_same_edge_simple(edge1, mapping, edge2)
        self.assertTrue(json_class.are_same_edge(edge1, mapping, edge2))

        mapping = {0:0,1:1}
        self.print_debug_are_same_edge_simple(edge1, mapping, edge2)
        self.assertTrue(json_class.are_same_edge(edge2, mapping, edge1))

        mapping ={0:1,0:1}
        self.print_debug_are_same_edge_simple(edge1, mapping, edge2)
        self.assertFalse(json_class.are_same_edge(edge2, mapping, edge1))


    def test_are_same_edge(self):
        self.mapping_dict = {}
        self.mapping_dict[0] = 1 # 0 -> 1
        self.mapping_dict[1] = 2 # 1 -> 2
        self.mapping_dict[2] = 3 # 2 -> 3

        edge1 = {"to"  : 0,"from": 1}
        edge2 = {"to"  : 0,"from": 1}

        self.assertFalse(json_class.are_same_edge(edge1, self.mapping_dict, edge2))

        edge1 = {"to"  : 0,"from": 1}
        edge2 = {"to"  : 1,"from": 2}
        self.assertTrue(json_class.are_same_edge(edge1, self.mapping_dict, edge2))

    def test_inclusion_json(self):
        json_imported = self.JSON_class.import_json(self.test_file_path / 'mini_baseline.json')
        self.JSON_class.json_to_export = json_imported
        json_imported = self.JSON_class.import_json(self.test_file_path / 'mini_baseline_to_match.json')

        self.mapping_dict = json_class.create_node_mapping(self.JSON_class.json_to_export, json_imported)
        wrong = json_class.is_graphe_included(self.JSON_class.json_to_export, self.mapping_dict, json_imported)
        self.assertEqual(wrong,0)

        tmp_modified = self.JSON_class.json_to_export
        tmp_modified["edges"][0]["to"] = 1
        tmp_modified["edges"][0]["from"] = 0
        self.JSON_class.json_to_export = tmp_modified
        wrong = json_class.is_graphe_included(self.JSON_class.json_to_export, self.mapping_dict, json_imported)
        self.assertEqual(wrong,1)

    def test_matching_graphe_percentage(self):
        json_imported = self.JSON_class.import_json(self.test_file_path / 'mini_baseline.json')
        self.JSON_class.json_to_export = json_imported
        json_imported = self.JSON_class.import_json(self.test_file_path / 'mini_baseline_to_match.json')

        self.assertEqual(json_class.matching_graphe_percentage(self.JSON_class.json_to_export, json_imported), 1)
        self.assertEqual(json_class.matching_graphe_percentage(json_imported, self.JSON_class.json_to_export), 1)

        tmp_modified = self.JSON_class.json_to_export
        tmp_modified["edges"][0]["to"] = 1
        tmp_modified["edges"][0]["from"] = 0
        self.JSON_class.json_to_export = tmp_modified
        self.assertAlmostEqual(json_class.matching_graphe_percentage(self.JSON_class.json_to_export, json_imported), 0.66, delta=0.01)


if __name__ == '__main__':
    unittest.main()