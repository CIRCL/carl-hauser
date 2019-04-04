# ==================== ------------------------ ====================
#                       Configuration launcher
# STD imports
import logging
import pathlib
import json

# Own imports
import utility_lib.filesystem_lib as filesystem_lib
import utility_lib.json_class as json_class
import utility_lib.graph_lib as graph_lib
import configuration
import ImageHash.imagehash_test as image_hash
import TLSH.tlsh_test as tlsh
import OpenCV.opencv as opencv


class Configuration_launcher():
    def __init__(self,
                 source_pictures_dir: pathlib.Path,
                 output_folder: pathlib.Path,
                 ground_truth_json: pathlib.Path,

                 img_type: configuration.SUPPORTED_IMAGE_TYPE):
        logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.source_pictures_dir = source_pictures_dir
        self.output_folder = output_folder
        self.ground_truth_json = ground_truth_json
        self.img_type = img_type

        self.logger.warning("Creation of filesystem handler : deletion of 0-sized pictures in source folder.")
        tmp_conf = configuration.Default_configuration()
        tmp_conf.IMG_TYPE = img_type
        self.filesystem_handler = filesystem_lib.File_System(conf=tmp_conf)
        self.filesystem_handler.clean_folder(self.source_pictures_dir)

    def auto_launch(self):
        self.logger.info("==== ----- LAUNCHING AUTO CONF LAUNCHER ---- ==== ")
        self.auto_launch_image_hash()
        self.auto_launch_tlsh()
        self.auto_launch_orb()

    def auto_launch_image_hash(self):
        self.logger.info("==== ----- LAUNCHING IMAGE HASH ALGOS ---- ==== ")

        # Create conf
        curr_configuration = configuration.Default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        curr_configuration.SAVE_PICTURE = False
        curr_configuration.OUTPUT_DIR = self.output_folder

        list_to_execute = [configuration.ALGO_TYPE.A_HASH,
                           configuration.ALGO_TYPE.P_HASH,
                           configuration.ALGO_TYPE.P_HASH_SIMPLE,
                           configuration.ALGO_TYPE.D_HASH,
                           configuration.ALGO_TYPE.D_HASH_VERTICAL,
                           configuration.ALGO_TYPE.W_HASH]

        # Launch
        for type in list_to_execute:
            curr_configuration.ALGO = type
            curr_configuration.OUTPUT_DIR = self.output_folder / image_hash.Image_hash_execution_handler.conf_to_string(curr_configuration)
            try:
                eh = image_hash.Image_hash_execution_handler(conf=curr_configuration)
                eh.do_full_test()
            except Exception as e:
                logging.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")

    def auto_launch_tlsh(self):
        self.logger.info("==== ----- LAUNCHING TLSH algos ---- ==== ")

        # Create conf
        curr_configuration = configuration.Default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        curr_configuration.SAVE_PICTURE = False
        curr_configuration.OUTPUT_DIR = self.output_folder

        # Launch
        list_to_execute = [configuration.ALGO_TYPE.TLSH,
                           configuration.ALGO_TYPE.TLSH_NO_LENGTH]

        # Launch
        for type in list_to_execute:
            curr_configuration.ALGO = type
            curr_configuration.OUTPUT_DIR = self.output_folder / tlsh.TLSH_execution_handler.conf_to_string(curr_configuration)
            try:
                eh = tlsh.TLSH_execution_handler(conf=curr_configuration)
                eh.do_full_test()
            except Exception as e:
                logging.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")

    def auto_launch_orb(self):
        self.logger.info("==== ----- LAUNCHING ORB algos ---- ==== ")

        # Create conf
        curr_configuration = configuration.ORB_default_configuration()
        curr_configuration.SOURCE_DIR = self.source_pictures_dir
        curr_configuration.GROUND_TRUTH_PATH = self.ground_truth_json
        curr_configuration.IMG_TYPE = self.img_type
        curr_configuration.SAVE_PICTURE = False
        curr_configuration.OUTPUT_DIR = self.output_folder

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
                                eh = opencv.OpenCV_execution_handler(conf=curr_configuration)
                                eh.do_full_test()
                            except Exception as e:
                                logging.error(f"Aborting this configuration. Current configuration thrown an error : {e} ")

    @staticmethod
    def create_tldr(folder: pathlib.Path, output_file: pathlib.Path):

        f = open(str(output_file.resolve()), "a+")  # Append and create if does not exist

        global_list = []
        for x in folder.resolve().iterdir():

            global_txt = ""

            if x.is_dir():
                global_txt += (x.name).ljust(95, " ") + "\t"
                stat_file = x / "stats.txt"
                if stat_file.exists():

                    with open(str(stat_file.resolve())) as json_file:
                        json_file = str(json_file.read()).replace("'", '"')
                        data = json.loads(json_file)
                        LEN = 34
                        global_txt += ("TRUE_POSITIVE = " + str(data["TRUE_POSITIVE_RATE"])).ljust(LEN, " ") + " \t"
                        global_txt += ("PRE_COMPUTING = " + str(data["TIME_PER_PICTURE_PRE_COMPUTING"])).ljust(LEN, " ") + " \t"
                        global_txt += ("MATCHING = " + str(data["TIME_PER_PICTURE_MATCHING"])).ljust(LEN, " ")

                        global_list.append([global_txt, data["TRUE_POSITIVE_RATE"]])

                else:
                    global_txt += "NO RESULT / ERROR"

                    global_list.append([global_txt, -1])

        global_list = sorted(global_list, key=lambda l: l[1], reverse=True)

        for x in global_list:
            f.write(x[0] + "\r\n")
        f.close()

        logger = logging.getLogger(__name__)
        logger.info("Overview written")

    @staticmethod
    def create_similarity_matrix(folder: pathlib.Path, output_file: pathlib.Path):
        global_result = Configuration_launcher.create_inclusion_matrix(folder=folder)
        Configuration_launcher.save_similarity_json(global_result, output_file.with_suffix(".json"))

        ordo, absi, values = Configuration_launcher.inclusion_matrix_to_triple_array(global_result)

        graph = graph_lib.Graph_handler()
        graph.set_values(ordo, absi, values)

        graph.save_matrix(output_file.with_suffix(".png"))

    @staticmethod
    def get_graph_list(folder: pathlib.Path):
        graphe_list = []

        # For all graphe
        for x in folder.resolve().iterdir():
            if x.is_dir():
                curr_graphe_file = x / "graphe.json"
                if curr_graphe_file.is_file():
                    # We have a valid graphe file to load
                    with open(str(curr_graphe_file.resolve())) as json_file:
                        json_file = str(json_file.read()).replace("'", '"')
                        data = json.loads(json_file)

                        # Load each graphe
                        graphe_list.append([x.name, data])

        return graphe_list

    @staticmethod
    def create_pairing_matrix(folder: pathlib.Path, truth_file: pathlib.Path, output_folder: pathlib.Path):
        graphe_list = Configuration_launcher.get_graph_list(folder)

        logger = logging.getLogger(__name__)
        logger.info("Pairing matrix written")

    @staticmethod
    def create_inclusion_matrix(folder: pathlib.Path):
        # TODO : safe path in case it's not a Pathlib : is it necessary ?
        # folder = filesystem_lib.File_System.safe_path(folder)
        # output_folder = filesystem_lib.File_System.safe_path(output_folder)
        logger = logging.getLogger(__name__)
        logger.info(f"Creating inclusion matrix for {folder}")

        graphe_list = Configuration_launcher.get_graph_list(folder)

        # For all graphe A
        global_result = []
        for curr_graphe_a in graphe_list:
            logger.debug(f"Checking graphe {curr_graphe_a[0]} ...")

            tmp_result_graph_a = {}
            tmp_result_graph_a["source"] = curr_graphe_a[0]

            tmp_result_graph_b = []
            # For all graphe B
            for curr_graphe_b in graphe_list:
                logger.debug(f"Checking graphe {curr_graphe_a[0]} with {curr_graphe_b[0]}")

                # Evaluate each graphe A inclusion to each other graphe B
                tmp_mapping_dict = json_class.create_node_mapping(curr_graphe_a[1], curr_graphe_b[1])
                wrong_edge_list = json_class.is_graphe_included(curr_graphe_a[1], tmp_mapping_dict, curr_graphe_b[1])

                # Compute similarity based on inclusion (card(inclusion)/card(source))
                nb_edges = len(curr_graphe_a[1]["edges"])
                curr_similarity = 1 - (len(wrong_edge_list) / nb_edges)

                # Store the similarity in an array
                tmp_dict = {}
                tmp_dict["compared_to"] = curr_graphe_b[0]
                tmp_dict["similarity"] = curr_similarity
                tmp_result_graph_b.append(tmp_dict)

            # Store the similarity array as json
            # TODO : worth to sort it ? Would impact next computation
            tmp_result_graph_b = sorted(tmp_result_graph_b, key=lambda l: l["similarity"], reverse=True)
            tmp_result_graph_a["similar_to"] = tmp_result_graph_b
            global_result.append(tmp_result_graph_a)

        # Alphabetical order
        global_result = sorted(global_result, key=lambda l: len(l["source"]))

        return global_result

    @staticmethod
    def save_similarity_json(similarity_matrix, output_file: pathlib.Path):
        # Store the similarity array as picture
        # output_file = output_folder / "inclusion_matrix.json"
        f = open(str(output_file.resolve()), "w+")  # Overwrite and create if does not exist
        tmp_json = json.dumps(similarity_matrix)
        f.write(tmp_json)
        f.close()

        logger = logging.getLogger(__name__)
        logger.info("Inclusion matrix written")

        return output_file

    @staticmethod
    def inclusion_matrix_to_triple_array(inclusion_dict):
        ordo, absi, values = [], [], []

        for curr_source in inclusion_dict:
            # For the axis
            ordo.append(curr_source["source"])
            absi.append(curr_source["source"])

        for curr_source in ordo:
            tmp_row_values = []
            for curr_target in absi:
                tmp_similarity_list = Configuration_launcher.find_source_in_list(inclusion_dict, "source", curr_source)["similar_to"]
                value = Configuration_launcher.find_source_in_list(tmp_similarity_list, "compared_to", curr_target)["similarity"]
                tmp_row_values.append(value)
            values.append(tmp_row_values)

        return ordo, absi, values

    @staticmethod
    def find_source_in_list(list, tag, to_find):
        for x in list:
            if x[tag] == to_find:
                return x
        else:
            return None


if __name__ == '__main__':

    # =============================
    source_pictures_dir = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing/")
    output_folder = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_output/")
    ground_truth_json = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing.json")
    output_file = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_output.overview")
    output_similarity_matrix = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_output.matrix")
    '''
    img_type = configuration.SUPPORTED_IMAGE_TYPE.PNG

    config_launcher = Configuration_launcher(source_pictures_dir=source_pictures_dir.resolve(),
                                             output_folder=output_folder.resolve(),
                                             ground_truth_json=ground_truth_json.resolve(),
                                             img_type=img_type)
    config_launcher.auto_launch()
    '''
    Configuration_launcher.create_tldr(folder=output_folder, output_file=output_file)
    Configuration_launcher.create_similarity_matrix(folder=output_folder, output_file=output_similarity_matrix)

    # =============================
    source_pictures_dir = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_bmp/")
    output_folder = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_bmp_output/")
    ground_truth_json = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_bmp.json")
    output_file = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_bmp_output.overview")
    output_similarity_matrix = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_bmp_output.matrix")

    '''
    img_type = configuration.SUPPORTED_IMAGE_TYPE.BMP

    config_launcher = Configuration_launcher(source_pictures_dir=source_pictures_dir.resolve(),
                                             output_folder=output_folder.resolve(),
                                             ground_truth_json=ground_truth_json.resolve(),
                                             img_type=img_type)
    config_launcher.auto_launch()
    '''
    Configuration_launcher.create_tldr(folder=output_folder, output_file=output_file)
    Configuration_launcher.create_similarity_matrix(folder=output_folder, output_file=output_similarity_matrix)

    # =============================
    source_pictures_dir = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_COLORED/")
    output_folder = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_COLORED_output/")
    ground_truth_json = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing.json")
    output_file = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_COLORED_output.overview")
    output_similarity_matrix = pathlib.Path.cwd() / pathlib.Path("../datasets/raw_phishing_COLORED_output.matrix")

    '''
    img_type = configuration.SUPPORTED_IMAGE_TYPE.PNG

    config_launcher = Configuration_launcher(source_pictures_dir=source_pictures_dir.resolve(),
                                             output_folder=output_folder.resolve(),
                                             ground_truth_json=ground_truth_json.resolve(),
                                             img_type=img_type)
    config_launcher.auto_launch()
    '''

    Configuration_launcher.create_tldr(folder=output_folder, output_file=output_file)
    Configuration_launcher.create_similarity_matrix(folder=output_folder, output_file=output_similarity_matrix)
