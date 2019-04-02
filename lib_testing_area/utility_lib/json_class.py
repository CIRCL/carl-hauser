from .picture_class import Picture
from typing import List
import json
import pathlib

TOP_K_EDGE = 1

class JSON_GRAPHE() :
    def __init__(self, json_value={}):
        self.json_to_export = json_value
        self.quality = None

    # =========================== -------------------------- ===========================
    #                                   IMPORT / EXPORT

    @staticmethod
    def import_json(file_path):
        with file_path.open() as data_file:
            json_imported = json.load(data_file)

        return json_imported

    def json_export(self, file_name='test.json'):
        with open(pathlib.Path(file_name), 'w') as outfile:
            json.dump(self.json_to_export, outfile)

    # =========================== -------------------------- ===========================
    #                                GRAPHE MODIFICATION

    def json_add_nodes(self, picture_list : List[Picture]) :
        nodes_list = []

        # Add all nodes
        for curr_picture in picture_list :
            nodes_list.append(curr_picture.to_node_json_object())

        self.json_to_export["nodes"] = nodes_list

        return self

    def json_add_top_matches(self, json_tmp, sorted_picture_list : List[Picture], target_picture : Picture, k_edge=TOP_K_EDGE) :
        # Preprocess to remove target picture from matches
        offset = remove_target_picture_from_matches(sorted_picture_list,target_picture)

        # Get current list of matches
        edges_list = json_tmp.get("edges", [])

        # Add all edges with labels
        for i in range(0,min(len(sorted_picture_list),k_edge)) :
            tmp_obj = {}
            tmp_obj["from"] = target_picture.id
            tmp_obj["to"] = sorted_picture_list[i+offset].id
            tmp_obj["label"] = "rank " + str(i) + "(" + str(sorted_picture_list[i+offset].distance) + ")"
            edges_list.append(tmp_obj)

        # Store in JSON variable
        json_tmp["edges"] = edges_list

        return json_tmp

    def evaluate_json(self, baseline_path):
        json_imported = self.import_json(baseline_path)
        self.quality = matching_graphe_percentage(self.json_to_export, json_imported)

        self.json_to_export["quality"] =  self.quality

        return self, self.quality

    def replace_type(selfs, json, target_type='.bmp'):
        nodes = json["nodes"]

        # Convert the extension of images in a graphe json file
        for curr_node in nodes :
            curr_node["image"] = str(pathlib.Path(curr_node["image"]).with_suffix(target_type))

        json["nodes"] = nodes
        return json

# =========================== -------------------------- ===========================
 #                                MISC TOOLS

def remove_target_picture_from_matches(sorted_picture_list : List[Picture], target_picture : Picture):
    offset = 0
    if sorted_picture_list != [] and target_picture.is_same_picture_as(sorted_picture_list[0]):
        # If first picture is the original picture we skip.
        print("Removed first choice : " + sorted_picture_list[0].path.name)
        offset += 1

    print(f"Offset after target picture removal from matches : {offset}")

    return offset

# =========================== -------------------------- ===========================
 #                                GRAPHE TESTS

def matching_graphe_percentage(candidate_graphe, ground_truth_graphe):
    mapping_dict = create_node_mapping(candidate_graphe, ground_truth_graphe)
    wrong = is_graphe_included(candidate_graphe, mapping_dict, ground_truth_graphe)

    edges_length = len(candidate_graphe["edges"])

    return 1 - wrong/edges_length

def create_node_mapping(candidate_graphe, ground_truth_graphe):
    mapping_dict = {}

    candidate_nodes = candidate_graphe["nodes"]
    ground_truth_nodes = ground_truth_graphe["nodes"]

    # For all pictures of the output, give the matching picture in the ground truth dictionnary
    for curr_picture in candidate_nodes:
        for candidate_picture in ground_truth_nodes:

            if curr_picture["image"] == candidate_picture["image"] :
                mapping_dict[curr_picture['id']] = candidate_picture["id"] # .append(curr_picture.to_node_json_object())
                continue

    return mapping_dict

def is_graphe_included(candidate_graphe, mapping_dict, ground_truth_graphe):
    '''
    Answer the question : "Does all candidate graphes edges are included in the ground truth graphe edges ?"
    :param candidate_graphe:
    :param mapping_dict:
    :param ground_truth_graphe:
    :return:
    '''

    candidate_edges = candidate_graphe["edges"]
    ground_truth_edges = ground_truth_graphe["edges"]

    wrong = 0

    # For all pictures of the output, give the matching picture in the ground truth dictionnary
    for curr_candidate_edge in candidate_edges:
        found = False
        for truth_edge in ground_truth_edges:
            if are_same_edge(curr_candidate_edge,mapping_dict,truth_edge) :
                found = True
                continue
        if not found :
            print(f"Edge : {str(curr_candidate_edge)} not found in baseline graph.")
            wrong += 1

    return wrong

def are_same_edge(edge1, matching, edge2):
    try :
        if matching[edge1["to"]] == edge2["to"] and matching[edge1["from"]] == edge2["from"] :
            return True
    except KeyError as e :
        print("JSON_CLASS : MATCHING AND EDGES ARE NOT CONSISTENT : a source edge index is not part of the matching" + str(e))

    return False


