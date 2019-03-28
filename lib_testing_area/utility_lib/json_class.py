from .picture_class import Picture
from typing import List
import json
import pathlib

TOP_K_EDGE = 1

class JSON_VISUALISATION() :
    def __init__(self, json_value={}):
        self.json_to_export = json_value

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

    def json_export(self, file_name='test.json'):
        with open(pathlib.Path(file_name), 'w') as outfile:
            json.dump(self.json_to_export, outfile)

def remove_target_picture_from_matches(sorted_picture_list : List[Picture], target_picture : Picture):
    offset = 0
    if sorted_picture_list != [] and target_picture.is_same_picture_as(sorted_picture_list[0]):
        # If first picture is the original picture we skip.
        print("Removed first choice : " + sorted_picture_list[0].path.name)
        offset += 1

    print(f"Offset after target picture removal from matches : {offset}")

    return offset