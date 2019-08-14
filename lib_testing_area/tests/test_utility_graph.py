# -*- coding: utf-8 -*-

from .context import *

import unittest

class test_template(unittest.TestCase):
    """Basic test cases."""

    def setUp(self):
        self.logger = logging.getLogger()
        self.conf = configuration.Default_configuration()
        self.test_file_path = pathlib.Path.cwd() / pathlib.Path("tests/test_files/utility/graph/")
        self.result_folder_path = self.test_file_path / 'raw_results'
        self.baseline_path = self.test_file_path / 'baseline' / "graphe.py"

    def test_absolute_truth_and_meaning(self):
        self.assertTrue(True)


    def test_writing_similarity_json(self):
        global_result = graph_lib.Graph_handler.create_inclusion_matrix(folder=self.result_folder_path)
        output_file = graph_lib.Graph_handler.save_matrix_to_json(global_result, self.test_file_path / "similarity_test.json")

        self.assertEqual(output_file.exists(), True)

    def test_similarity_matrix(self):
        global_result = graph_lib.Graph_handler.create_inclusion_matrix(folder=self.result_folder_path)

        results_1 = graph_lib.Graph_handler.find_source_in_list(global_result, "source", "results_1")["similar_to"]
        results_2 = graph_lib.Graph_handler.find_source_in_list(global_result, "source", "results_2")["similar_to"]
        results_3 = graph_lib.Graph_handler.find_source_in_list(global_result, "source", "results_3")["similar_to"]

        # Comparing graphe1 to graph1, then to graph2, then to graph3 ...
        self.assertAlmostEqual(graph_lib.Graph_handler.find_source_in_list(results_1, "compared_to", "results_1")["similarity"], 1, delta=0.000001)
        self.assertAlmostEqual(graph_lib.Graph_handler.find_source_in_list(results_1, "compared_to", "results_2")["similarity"], 0, delta=0.000001)
        self.assertAlmostEqual(graph_lib.Graph_handler.find_source_in_list(results_1, "compared_to", "results_3")["similarity"], 0.33333, delta=0.01)

        # Comparing graphe2 to graph1, then to graph2, then to graph3 ...
        self.assertAlmostEqual(graph_lib.Graph_handler.find_source_in_list(results_2, "compared_to", "results_1")["similarity"], 0, delta=0.000001)
        self.assertAlmostEqual(graph_lib.Graph_handler.find_source_in_list(results_2, "compared_to", "results_2")["similarity"], 1, delta=0.000001)
        self.assertAlmostEqual(graph_lib.Graph_handler.find_source_in_list(results_2, "compared_to", "results_3")["similarity"], 0.5, delta=0.000001)

        # Comparing graphe3 to graph1, then to graph2, then to graph3 ...
        self.assertAlmostEqual(graph_lib.Graph_handler.find_source_in_list(results_3, "compared_to", "results_1")["similarity"], 0.5, delta=0.000001)
        self.assertAlmostEqual(graph_lib.Graph_handler.find_source_in_list(results_3, "compared_to", "results_2")["similarity"], 0.5, delta=0.000001)
        self.assertAlmostEqual(graph_lib.Graph_handler.find_source_in_list(results_3, "compared_to", "results_3")["similarity"], 1, delta=0.000001)

    def test_similarity_to_triple_array(self):
        global_result = graph_lib.Graph_handler.create_inclusion_matrix(folder=self.result_folder_path)

        ordo, absi, values = graph_lib.Graph_handler.inclusion_matrix_to_triple_array(global_result)

        self.assertEqual(ordo[0], "results_1")
        self.assertEqual(ordo[1], "results_2")
        self.assertEqual(ordo[2], "results_3")

        self.assertEqual(absi[0], "results_1")
        self.assertEqual(absi[1], "results_2")
        self.assertEqual(absi[2], "results_3")

        self.assertAlmostEqual(values[0][0], 1)
        self.assertAlmostEqual(values[0][1], 0)
        self.assertAlmostEqual(values[0][2], 0.333, delta=0.001)

        self.assertAlmostEqual(values[1][0], 0)
        self.assertAlmostEqual(values[1][1], 1)
        self.assertAlmostEqual(values[1][2], 0.5)

        self.assertAlmostEqual(values[2][0], 0.5)
        self.assertAlmostEqual(values[2][1], 0.5)
        self.assertAlmostEqual(values[2][2], 1)

    def manual_test_similarity_matrix_printing(self):
        global_result = graph_lib.Graph_handler.create_inclusion_matrix(folder=self.result_folder_path)
        ordo, absi, values = graph_lib.Graph_handler.inclusion_matrix_to_triple_array(global_result)

        graph = graph_lib.Graph_handler()
        graph.set_values(ordo, absi, values)

        graph.show_matrix()

    def test_similarity_matrix_saving(self):
        global_result = graph_lib.Graph_handler.create_inclusion_matrix(folder=self.result_folder_path)
        ordo, absi, values = graph_lib.Graph_handler.inclusion_matrix_to_triple_array(global_result)

        graph = graph_lib.Graph_handler()
        graph.set_values(ordo, absi, values)

        graph.save_matrix(self.test_file_path / "similarity_test" / "matrix.png")

    def test_get_graph_list(self):
        tmp_gl = graph_lib.Graph_handler.get_graph_list(self.result_folder_path)
        pprint.pprint(tmp_gl)

if __name__ == '__main__':
    unittest.main()