from .picture_class import Picture
from typing import List
import json
import pathlib
import logging

import configuration
import results
import utility_lib.filesystem_lib as filesystem_lib
import utility_lib.graph_lib as graph_lib
import matplotlib.pyplot as plt

TOP_K_EDGE = 1
MULT_FACTOR_VALUE = 10  # See : http://visjs.org/docs/network/edges.html (Min 1, MAX 15 so min 0, max 10)


class Json_handler():
    def __init__(self, conf: configuration.Default_configuration):
        self.conf = conf
        self.graphe = graph_lib.Graphe()
        self.quality = None

    # =========================== -------------------------- ===========================
    #                                   IMPORT / EXPORT
    @staticmethod
    def import_json(file_path):
        json_imported = filesystem_lib.File_System.load_json(file_path)

        '''
        with file_path.open() as data_file:
            json_imported = json.load(data_file)
        '''

        return json_imported

    def json_export(self):
        file_out = self.conf.OUTPUT_DIR / "graphe.py"
        filesystem_lib.File_System.save_json(self.graphe, file_path=file_out)
        '''
        with open(pathlib.Path(file_out), 'w') as outfile:
        json.dump(self.json_to_export, outfile)
        '''

    # =========================== -------------------------- ===========================
    #                                GRAPHE MODIFICATION
    def json_add_nodes(self, picture_list: List[Picture]):
        nodes_list = []

        # Add all nodes
        for curr_picture in picture_list:
            nodes_list.append(curr_picture.to_node_json_object())

        self.graphe.nodes = nodes_list

        return self

    def json_add_top_matches(self, sorted_picture_list: List[Picture], target_picture: Picture, k_edge=TOP_K_EDGE):
        # Preprocess to remove target picture from matches
        offset = remove_target_picture_from_matches(sorted_picture_list, target_picture)

        # Get current list of matches
        edges_list = self.graphe.edges

        # Add all edges with labels
        for i in range(0, min(len(sorted_picture_list), k_edge)):
            tmp_obj = {}
            tmp_obj["from"] = target_picture.id
            tmp_obj["to"] = sorted_picture_list[i + offset].id
            tmp_obj["label"] = "rank " + str(i) + " (" + str(round(sorted_picture_list[i + offset].distance, 5)) + ")"
            edges_list.append(tmp_obj)
            # tmp_obj["value"] = str(sorted_picture_list[i+offset].distance * MULT_FACTOR_VALUE)
            tmp_obj["value"] = str(Json_handler.mapFromTo(sorted_picture_list[i + offset].distance, 0, 1, 20, 1))
            # TODO : store per algo
            tmp_obj["title"] = sorted_picture_list[i + offset].distance

        # Store in JSON variable
        self.graphe.edges = edges_list

        return self

    # =========================== -------------------------- ===========================
    #                        EVALUATION AND THRESHOLD COMPUTATION

    def evaluate_json(self, ground_truth_path: pathlib.PosixPath, results: results.RESULTS):

        # Load groundtruth graphe
        json_imported = self.import_json(ground_truth_path)
        imported_graphe = graph_lib.Graphe()
        imported_graphe.load_from_json(json_imported)

        mapping_dict = create_node_mapping(self.graphe, imported_graphe)

        # Evaluated
        results.TRUE_POSITIVE_RATE = matching_graphe_percentage(self.graphe, imported_graphe)
        results.COMPUTED_THREESHOLD = self.find_threeshold(self.graphe, mapping_dict, imported_graphe)

        # TODO : Review following function. Does not operate as they should ?
        threesholded_graphe = self.get_thresholded_graphe(self.graphe,results.COMPUTED_THREESHOLD)
        results.TRUE_POSITIVE_RATE_THREESHOLD = matching_graphe_percentage(threesholded_graphe,imported_graphe)

        return self, results

    def find_threeshold(self, candidate_graphe, mapping_dict, ground_truth_graphe):

        wrong_edges = is_graphe_included(candidate_graphe, mapping_dict, ground_truth_graphe)
        wrong_edges = sorted(wrong_edges, key=lambda x: x["title"])

        mode = self.conf.THREESHOLD_EVALUATION

        if mode == configuration.THRESHOLD_MODE.MIN_WRONG :
            # MIN :
            threshold = wrong_edges[0]["title"]
        elif mode == configuration.THRESHOLD_MODE.MAX_WRONG :
            # MAX :
            threshold = wrong_edges[len(wrong_edges)-1]["title"]
        elif mode == configuration.THRESHOLD_MODE.MEDIAN_WRONG :
            # Median :
            threshold = wrong_edges[len(wrong_edges)//2]["title"]
        elif mode == configuration.THRESHOLD_MODE.MAXIMIZE_TRUE_POSITIVE :
            threshold = self.find_best_threeshold(candidate_graphe.edges, mapping_dict, wrong_edges, self.conf.OUTPUT_DIR / "threshold.png")
        else :
            raise Exception("Incorrect mode for threshold finder.")

        return threshold


    def find_best_threeshold(self, edge_list, mapping_dict, wrong_edge_list, output_graphe: pathlib.Path):
        total_edge_list = edge_list.copy()
        sorted_edge_list = sorted(edge_list, key=lambda x: x["title"], reverse=True)

        wrong_length = len(wrong_edge_list)
        edges_length = len(sorted_edge_list)  +1

        threshold_list = []
        score_list = []
        # We simulate the removal of each edge, sorted by "bad to good"
        for curr_edge in sorted_edge_list :
            # Check if it's a bad edge
            for curr_bad_edge in wrong_edge_list :
                if are_same_edge(curr_edge, mapping_dict, curr_bad_edge):
                    total_edge_list.append([curr_edge, "bad"])
                    wrong_length -= 1

            edges_length -= 1
            score_list.append(1 - (wrong_length/edges_length))
            threshold_list.append(curr_edge["title"])

        # order :  X followed by Y
        plt.plot(threshold_list, score_list)
        plt.legend(('True-positive'), loc='upper right')
        plt.xlabel("Distance threshold applied to prune edges (arbitrary distance unit)")
        plt.ylabel("True-Positive score with the given threshold \n(% true positive within original edges)")
        plt.title("True-Positive rate regarding distance threshold")
        # plt.show()
        plt.savefig(str(output_graphe.resolve()))
        plt.clf()
        plt.cla()
        plt.close()

        return sorted_edge_list[score_list.index(max(score_list))]["title"]


    def get_thresholded_graphe(self, candidate_graphe, threshold):
        modified_graphe = graph_lib.Graphe()
        modified_graphe.nodes = candidate_graphe.nodes

        good = []

        # For all candidate edge
        for curr_edge in candidate_graphe.edges:
            if curr_edge["title"] >= threshold:
                good.append(curr_edge)

        modified_graphe.edges = good

        return modified_graphe

    # =========================== -------------------------- ===========================
    #                                     UTILITY

    def replace_type(selfs, json, target_type='.bmp'):
        nodes = json["nodes"]

        # Convert the extension of images in a graphe json file
        for curr_node in nodes:
            curr_node["image"] = str(pathlib.Path(curr_node["image"]).with_suffix(target_type))

        json["nodes"] = nodes
        return json

    @staticmethod
    def mapFromTo(x, a, b, c, d):
        y = (x - a) / (b - a) * (d - c) + c
        return y

# =========================== -------------------------- ===========================
#                                MISC TOOLS

def remove_target_picture_from_matches(sorted_picture_list: List[Picture], target_picture: Picture):
    offset = 0
    logger = logging.getLogger(__name__)

    if sorted_picture_list != [] and target_picture.is_same_picture_as(sorted_picture_list[0]):
        # If first picture is the original picture we skip.
        logger.debug("Removed first choice : " + sorted_picture_list[0].path.name)
        offset += 1

    logger.debug(f"Offset after target picture removal from matches : {offset}")

    return offset


# =========================== -------------------------- ===========================
#                                GRAPHE TESTS

def matching_graphe_percentage(candidate_graphe, ground_truth_graphe):
    mapping_dict = create_node_mapping(candidate_graphe, ground_truth_graphe)
    wrong = is_graphe_included(candidate_graphe, mapping_dict, ground_truth_graphe)

    edges_length = len(candidate_graphe.edges)
    wrong_length = len(wrong)

    return 1 - wrong_length / edges_length


def create_node_mapping(candidate_graphe, ground_truth_graphe):
    '''
    Create a mapping (dictionnary) as : #Node in candidate graphe gives the #Node in the ground truth graph
    :param candidate_graphe:
    :param ground_truth_graphe:
    :return:
    '''
    mapping_dict = {}

    candidate_nodes = candidate_graphe.nodes
    ground_truth_nodes = ground_truth_graphe.nodes

    # For all pictures of the output, give the matching picture in the ground truth dictionnary
    for curr_picture in candidate_nodes:
        for candidate_picture in ground_truth_nodes:

            if curr_picture["image"] == candidate_picture["image"]:
                mapping_dict[curr_picture['id']] = candidate_picture["id"]  # .append(curr_picture.to_node_json_object())
                continue

    return mapping_dict


def is_graphe_included(candidate_graphe , mapping_dict, ground_truth_graphe):
    '''
    Answer the question : "Does all candidate graphes edges are included in the ground truth graphe edges ?"
    :param candidate_graphe:
    :param mapping_dict:
    :param ground_truth_graphe:
    :return:
    '''

    candidate_edges = candidate_graphe.edges
    ground_truth_edges = ground_truth_graphe.edges
    logger = logging.getLogger(__name__)

    wrong = []

    # For all candidate edge
    for curr_candidate_edge in candidate_edges:
        found = False
        # Check if we find the corresponding edge in the target edges list, given the node mapping
        for truth_edge in ground_truth_edges:
            if are_same_edge(curr_candidate_edge, mapping_dict, truth_edge):
                found = True
                continue
        if not found:
            logger.debug(f"Edge : {str(curr_candidate_edge)} not found in target graph.")
            wrong.append(curr_candidate_edge)

    return wrong


def are_same_edge(edge1, matching, edge2):
    logger = logging.getLogger(__name__)

    try:
        if matching[edge1["to"]] == edge2["to"] and matching[edge1["from"]] == edge2["from"]:
            return True
    except KeyError as e:
        logger.error("JSON_CLASS : MATCHING AND EDGES ARE NOT CONSISTENT : a source edge index is not part of the matching" + str(e))

    return False


def merge_graphes(graphe1, to_graphe2):
    '''
    Merge graphe 1 into graphe 2, and return a merged graphe (copy done during the process! Inputs are unchanged)

    :param graphe1:
    :param to_graphe2:
    :return:
    '''
    mapping_dict = create_node_mapping(graphe1, to_graphe2)

    future_graphe = {}

    if len(graphe1.nodes) != len(to_graphe2.nodes):
        logger = logging.getLogger(__name__)
        logger.error("Graphs to merge don't have the same number of nodes ! ")
        # TODO : probably a problem to handle here

    future_graphe.nodes = to_graphe2.nodes.copy()
    future_graphe.edges = to_graphe2.edges.copy()

    # Get edge information and translate it
    for curr_edge in graphe1.edges:
        tmp_future_edge = {}
        tmp_future_edge["from"] = mapping_dict[curr_edge["from"]]
        tmp_future_edge["to"] = mapping_dict[curr_edge["to"]]
        tmp_future_edge["label"] = curr_edge["label"]
        # Add the edge
        future_graphe.edges.append(tmp_future_edge)

    return future_graphe
