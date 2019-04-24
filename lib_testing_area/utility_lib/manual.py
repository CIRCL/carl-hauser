# ============================================= ------------------------------------ =============================================
#                                                           MANUAL LABOR
# Handle trucated image too
from PIL import Image, ImageFile
import pathlib
import utility_lib.json_class as json_class
import utility_lib.filesystem_lib as filesystem_lib
import configuration
import pprint
import cv2

import utility_lib.models.edge_detector.edge as deepEdgeDetector

DEFAULT_TARGET_DIR = pathlib.Path("../../datasets/raw_phishing_bmp/")
DEFAULT_BASELINE_PATH = pathlib.Path("../../datasets/raw_phishing.json")
BMP_TARGET_DIR = pathlib.Path("../../datasets/raw_phishing_bmp/")
BMP_BASELINE_PATH = pathlib.Path("../../datasets/raw_phishing_bmp.json")

ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_bmp_from_png():

    p = pathlib.Path(DEFAULT_TARGET_DIR).glob('**/*')
    files = [x for x in p if x.is_file()]

    print(f"files list : {files}")

    # Convert pictures
    for file in files :
        name_img_source = pathlib.Path(DEFAULT_TARGET_DIR) / file.name
        name_img_target = pathlib.Path(BMP_TARGET_DIR) / pathlib.Path(file.name).with_suffix('.bmp')

        print(name_img_source, name_img_target)

        img = Image.open(name_img_source)
        img.save(name_img_target)

    # Convert baseline file
    conf = configuration.Default_configuration()
    conf.OUTPUT_DIR = BMP_BASELINE_PATH

    json_handler = json_class.Json_handler(conf)
    png_json = json_handler.import_json(pathlib.Path(DEFAULT_BASELINE_PATH)) # PNG one
    bmp_json = json_handler.replace_type(png_json, ".bmp")

    json_handler.json_to_export = bmp_json
    json_handler.json_export()

def run_edge_detector_deep(source_dir : pathlib.Path, target_dir: pathlib.Path):
    ded = deepEdgeDetector.DeepEdgeDetector(pathlib.Path("./models/edge_detector/deploy.prototxt"), pathlib.Path("./models/edge_detector/hed_pretrained_bsds.caffemodel"))

    p = source_dir.glob('**/*')
    files = [x for x in p if x.is_file()]

    print(f"files list : {files}")

    # Convert pictures
    for file in files:
        name_img_source = source_dir / file.name
        name_img_target = target_dir / file.name

        ded.original_edgify_picture(name_img_source, name_img_target)


def run_edge_detector_canny(source_dir : pathlib.Path, target_dir: pathlib.Path):
    ded = deepEdgeDetector.DeepEdgeDetector(pathlib.Path("./models/edge_detector/deploy.prototxt"), pathlib.Path("./models/edge_detector/hed_pretrained_bsds.caffemodel"))

    p = source_dir.glob('**/*')
    files = [x for x in p if x.is_file()]

    print(f"files list : {files}")

    # Convert pictures
    for file in files:
        name_img_source = source_dir / file.name
        name_img_target = target_dir / file.name

        img = cv2.imread(str(name_img_source.resolve()), 0)
        edges = cv2.Canny(img, 100, 200)
        cv2.imwrite(str(name_img_target.resolve()), edges)



def get_max_score(path_best : pathlib.Path):
    data = filesystem_lib.File_System.load_json(path_best)

    node_dict = {}
    outliers_nodes = []

    for i in data["edges"]:
        node_dict[i["to"]] = 1
        node_dict[i["from"]] = 1

    for i in data["nodes"]:
        if i["id"] not in node_dict.keys():
            outliers_nodes.append(i)

    pprint.pprint(outliers_nodes)
    print(len(outliers_nodes))
    print(f"MAX SCORE : {manual_score(data['nodes'], outliers_nodes)}")

def manual_score(total_nodes, outliers_nodes):
    edges_length = len(total_nodes)
    wrong_length = len(outliers_nodes)

    return 1 - wrong_length/edges_length

if __name__ == '__main__':
    # create_bmp_from_png()

    # get_max_score(DEFAULT_BASELINE_PATH)

    # run_edge_detector_deep(pathlib.Path("../../datasets/raw_phishing/"),pathlib.Path("../../datasets/raw_phishing_EDGES_DEEP/"))
    # run_edge_detector_canny(pathlib.Path("../../datasets/raw_phishing/"),pathlib.Path("../../datasets/raw_phishing_EDGES_CANNY/"))