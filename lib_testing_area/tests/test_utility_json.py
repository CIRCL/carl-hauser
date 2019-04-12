# -*- coding: utf-8 -*-

from .context import *

import unittest
import logging

class test_template(unittest.TestCase):
    """Basic test cases."""

    def setUp(self):
        # logging = logging.getLogger()
        self.conf = configuration.Default_configuration()
        self.JSON_class = json_class.Json_handler(conf=self.conf)
        self.test_file_path = pathlib.Path.cwd() / pathlib.Path("tests/test_files/utility/json")
        self.merge_folder = self.test_file_path / 'merge_test'

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

        self.assertEqual(self.mapping_dict[0], 1)
        self.assertEqual(self.mapping_dict[1], 2)
        self.assertEqual(self.mapping_dict[2], 0)

    def test_are_same_edge_simple(self):
        edge1 = {"to"  : 0,"from": 1}
        edge2 = {"to"  : 0,"from": 1}

        mapping = {0:0,1:1}
        self.assertTrue(json_class.are_same_edge(edge1, mapping, edge2))

        mapping = {0:0,1:1}
        self.assertTrue(json_class.are_same_edge(edge2, mapping, edge1))

        mapping ={0:1,0:1}
        self.assertFalse(json_class.are_same_edge(edge2, mapping, edge1))

    def test_mapFromTo(self):
        self.assertEqual(json_class.Json_handler.mapFromTo(1,0,1,20,0),0)
        self.assertEqual(json_class.Json_handler.mapFromTo(0,0,1,20,0),20)
        self.assertEqual(json_class.Json_handler.mapFromTo(1,0,1,0,20),20)
        self.assertEqual(json_class.Json_handler.mapFromTo(0,0,1,0,20),0)

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
        self.assertEqual(len(wrong),0)

        tmp_modified = self.JSON_class.json_to_export
        tmp_modified["edges"][0]["to"] = 1
        tmp_modified["edges"][0]["from"] = 0
        self.JSON_class.json_to_export = tmp_modified
        wrong = json_class.is_graphe_included(self.JSON_class.json_to_export, self.mapping_dict, json_imported)
        self.assertEqual(len(wrong),1)

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

    def test_merge_graphe(self):
        with open(str((self.merge_folder / "graphe_1.json").resolve())) as json_file:
            json_file = str(json_file.read()).replace("'", '"')
            graphe1 = json.loads(json_file)
        with open(str((self.merge_folder / "graphe_2.json").resolve())) as json_file:
            json_file = str(json_file.read()).replace("'", '"')
            graphe2 = json.loads(json_file)

        merged_graphe = json_class.merge_graphes(graphe1, graphe2)
        self.assertEqual(len(graphe1["edges"]),3)
        self.assertEqual(len(graphe2["edges"]),2)
        self.assertEqual(len(merged_graphe["edges"]),5)

if __name__ == '__main__':
    unittest.main()