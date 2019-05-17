# ==================== ------------------------ ====================
#                       Configuration launcher
# STD imports
import logging
import pathlib
import argparse
import traceback
import pprint
import resource


from configuration_launcher import Configuration_launcher
import configuration


# Launch example :
# ./launcher.py -v -i ../datasets/raw_phishing -t PNG -tldr -tldr_latex -p -tldr_pair -im -pm -sp -Ov -ih -tlsh -orb -ob -void

def dir_path(path):
    if pathlib.Path(path).exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


# Arguments handling
parser = argparse.ArgumentParser()

essential = parser.add_argument_group('essential')
essential.add_argument("-i", "--input", dest="input_folder", type=dir_path, help="read pictures from this folder")
essential.add_argument("-t", "--type", dest="type", type=str, help="image type : PNG or BMP")
essential.add_argument("-gt", "--ground_truth", dest="ground_truth", type=dir_path,help="ground truth file to compute scores")
essential.add_argument("-o", "--output", dest="output_folder", type=dir_path,help="write execution results to this folder")

utilities = parser.add_argument_group('utilities')
utilities.add_argument("-v", "--verbosity", dest='verbose',help="increase output verbosity : v is INFO level, vv is DEBUG level, ..", action="count",default=0)
utilities.add_argument("-sp", "--save_pictures", dest='save_pictures',help="save_picture of algorithms outputs (top3, matches, ..)", action="store_true")
utilities.add_argument("-Ov", "--overwrite", dest='overwrite', help="overwrite existing output folder and results",action="store_true")

outputs_group = parser.add_argument_group('outputs')
outputs_group.add_argument("-ao", "--all-outputs", dest='all_outputs',help="Use all ouputs methods", action="store_true")
outputs_group.add_argument("-tldr", "--toolongdidntread", dest='tldr', help="create an overview of results",action="store_true")
outputs_group.add_argument("-tldr_latex", "--toolongdidntread_in_latex", dest='tldr_latex',help="create an overview of results in latex format (table)", action="store_true")
outputs_group.add_argument("-p", "--pair_results", dest='pair_results', help="pair algorithm results, evaluate and store them",action="store_true")
outputs_group.add_argument("-tldr_pair", "--toolongdidntread_for_pairs", dest='tldr_pairs',help="create an overview of results for pairs", action="store_true")
outputs_group.add_argument("-im", "--inclusion_matrix", dest='inclusion_matrix', help="create an inclusion matrix of results",action="store_true")
outputs_group.add_argument("-pm", "--inclusion_matrix_for_pairs", dest='inclusion_matrix_pairs',help="create an inclusion matrixof results for pairs", action="store_true")

group_algos = parser.add_argument_group('algorithms')
group_algos.add_argument("-aa", "--all-algorithms", dest='all_algos',help="Use all algorithms", action="store_true")
group_algos.add_argument("-ih", "--imagehash", dest='imagehash',help="use image hash algorithms (a-hash, p-hash, d-hash, w-hash ...)", action="store_true")
group_algos.add_argument("-tlsh", "--tlsh", dest='tlsh', help="use tlsh hash algorithms", action="store_true")
group_algos.add_argument("-orb", "--orb", dest='orb_normal', help="use orb algorithms", action="store_true")
group_algos.add_argument("-ob", "--orb_bow", dest='orb_bow', help="use orb BoW algorithms", action="store_true")
group_algos.add_argument("-void", "--void", dest='void', help="use a void algorithm for reference", action="store_true")

args = parser.parse_args()

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

if args.all_algos :
    args.imagehash = True
    args.tlsh = True
    args.orb_normal = True
    args.orb_bow = True
    args.void = True

if args.all_outputs :
    args.tldr = True
    args.tldr_latex = True
    args.pair_results = True
    args.tldr_pairs = True
    args.inclusion_matrix = True
    args.inclusion_matrix_pairs = True

# For profiling :
# import cProfile
def manual():
    logger = logging.getLogger()  # See : https://stackoverflow.com/questions/50714316/how-to-use-logging-getlogger-name-in-multiple-modules
    logger.setLevel(logging.INFO)

    # create console handler
    std_handler = logging.StreamHandler()
    logfile_handler = logging.FileHandler('./logfile.log', 'a')

    # Set log level
    std_handler.setLevel(logging.INFO)
    logfile_handler.setLevel(logging.INFO)

    # create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - + %(relativeCreated)d - %(name)s - %(levelname)s - %(message)s')

    std_handler.setFormatter(formatter)
    logfile_handler.setFormatter(formatter)

    # add the handler to the logger
    logger.addHandler(std_handler)
    logger.addHandler(logfile_handler)

    base_path = [["../datasets/raw_phishing", configuration.SUPPORTED_IMAGE_TYPE.PNG],
                 # ["../datasets/raw_phishing_bmp", configuration.SUPPORTED_IMAGE_TYPE.BMP],
                 # ["../datasets/raw_phishing_COLORED", configuration.SUPPORTED_IMAGE_TYPE.PNG],
                 # ["../datasets/raw_phishing_Tesseract", configuration.SUPPORTED_IMAGE_TYPE.PNG],
                 # ["../datasets/raw_phishing_EDGES_CANNY", configuration.SUPPORTED_IMAGE_TYPE.PNG],
                 # ["../datasets/raw_phishing_EDGES_DEEP", configuration.SUPPORTED_IMAGE_TYPE.PNG]
                 ]

    for curr_base_path, img_type in base_path:

        # =============================
        # Source folder with raw pictures
        source_pictures_dir = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "/")
        # Source ground truth json
        ground_truth_json = pathlib.Path.cwd() / pathlib.Path(curr_base_path + ".json")
        # Output folder for statistics of executions folders
        output_folder = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output/")
        output_folder.resolve().mkdir(exist_ok=True)

        # Outut folder for statics of executions for paired results
        paired_output_folder = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output_paired/")
        paired_output_folder.resolve().mkdir(exist_ok=True)

        # Output overview files
        output_overview_file = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output.overview")
        output_latex_overview_file = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output.latex.overview")
        output_overview_paired_file = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output_paired.overview")

        # Output matrix files
        output_similarity_matrix = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output.matrix")
        output_paired_matrix = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "_output_paired.matrix")

        # =============================
        # img_type = configuration.SUPPORTED_IMAGE_TYPE.PNG
        config_launcher = Configuration_launcher(source_pictures_dir=source_pictures_dir.resolve(),
                                                 output_folder=output_folder.resolve(),
                                                 ground_truth_json=ground_truth_json.resolve(),
                                                 img_type=img_type,
                                                 overwrite_folder=False,
                                                 args=args)
        # For profiling : cProfile.run("
        config_launcher.auto_launch()
        # ")

        # Create overview for simple results
        Configuration_launcher.create_tldr(folder=output_folder, output_file=output_overview_file)
        Configuration_launcher.create_latex_tldr(folder=output_folder, output_file=output_latex_overview_file)

        # Create overview for paired results
        Configuration_launcher.create_paired_results(input_folder=output_folder,target_pair_folder=paired_output_folder,ground_truth_json=ground_truth_json)
        Configuration_launcher.create_tldr(folder=paired_output_folder, output_file=output_overview_paired_file)

        # Create matrixes
        Configuration_launcher.create_and_export_inclusion_matrix(folder=output_folder,output_file=output_similarity_matrix)
        Configuration_launcher.create_and_export_pair_matrix(input_folder=output_folder,ground_truth_json=ground_truth_json,output_file=output_paired_matrix)


if __name__ == '__main__':
    pprint.pprint(args.__dict__)

    logger = logging.getLogger()  # See : https://stackoverflow.com/questions/50714316/how-to-use-logging-getlogger-name-in-multiple-modules
    logger.setLevel(logging.INFO)

    # create console handler
    std_handler = logging.StreamHandler()
    logfile_handler = logging.FileHandler('./logfile.log', 'w')

    # Set log level
    std_handler.setLevel(logging.INFO)
    logfile_handler.setLevel(logging.INFO)

    # create formatter and add it to the handler
    std_handler.setFormatter(configuration.FORMATTER)
    logfile_handler.setFormatter(configuration.FORMATTER)

    # add the handler to the logger
    logger.addHandler(std_handler)
    logger.addHandler(logfile_handler)

    # ================================ ---------------------- ================================
    #                                  Argument handling

    # Handle argument image type
    tmp_type = configuration.SUPPORTED_IMAGE_TYPE.PNG
    if args.type == "BMP":
        tmp_type = configuration.SUPPORTED_IMAGE_TYPE.BMP
    elif args.type not in ["PNG", "BMP"]:
        logger.warning("Image type not recognized. Chosen by default : PNG type")

    # Handle paths
    tmp_path = args.input_folder

    # Handle paths
    tmp_path_gt = args.ground_truth

    # Handle paths
    tmp_path_out = args.output_folder

    # TODO : Checks on the input folder ? Already handled by the check (see upper in this file)

    # ================================ ---------------------- ================================
    #                                           Launch

    base_path = [[tmp_path, tmp_type]]

    for curr_base_path, img_type in base_path:
        curr_base_path = curr_base_path
        # =============================
        # Source folder with raw pictures
        if tmp_path is not None:
            source_pictures_dir = pathlib.Path(tmp_path)
        else:
            logger.warning("Impossible to recognize input path. Default input path selected.")
            source_pictures_dir = pathlib.Path.cwd() / pathlib.Path(curr_base_path + "/")

        source_pictures_dir = source_pictures_dir.resolve()
        logger.info(f"Input path : {source_pictures_dir}")

        # Source ground truth json
        if tmp_path_gt is not None:
            ground_truth_json = pathlib.Path(tmp_path_gt)
        else:
            logger.warning("Impossible to recognize ground truth path. Default ground truth path selected.")
            ground_truth_json = pathlib.Path.cwd() / pathlib.Path(curr_base_path + ".json")

        ground_truth_json = ground_truth_json.resolve()
        logger.info(f"Ground truth path : {ground_truth_json}")

        curr_base_path = pathlib.Path(curr_base_path)

        if tmp_path_out is None:
            logger.warning("Impossible to recognize output path. Default output path selected.")
            # We don't have any correct output path
            base_path = pathlib.Path.cwd() / curr_base_path.parent  # /my/wonderful/output / ../INPUT/(dataset)
        else:
            # We have a correct output path
            if pathlib.Path(tmp_path_out).is_absolute():
                base_path = pathlib.Path(tmp_path_out)  # /my/wonderful/output
            else:
                base_path = pathlib.Path.cwd() / pathlib.Path(tmp_path_out)  # /my/exec/path / ../my/output/

        base_path = base_path.resolve()

        # Output folder for statistics of executions folders
        # /my/wonderful/output / ../INPUT/dataset_output/ OR /my/wonderful/output/dataset_output/ OR  my/exec/path / ../my/output/dataset_output/
        output_folder = base_path / pathlib.Path(curr_base_path.name + "_output/")
        if not output_folder.resolve().exists(): output_folder.resolve().mkdir()

        # Outut folder for statics of executions for paired results
        paired_output_folder = base_path / pathlib.Path(curr_base_path.name + "_output_paired/")
        if not paired_output_folder.resolve().exists(): paired_output_folder.resolve().mkdir()

        # Output overview files
        output_overview_file = base_path / pathlib.Path(curr_base_path.name + "_output.overview")
        output_latex_overview_file = base_path / pathlib.Path(curr_base_path.name + "_output.latex.overview")
        output_overview_paired_file = base_path / pathlib.Path(curr_base_path.name + "_output_paired.overview")

        # Output matrix files
        output_similarity_matrix = base_path / pathlib.Path(curr_base_path.name + "_output.matrix")
        output_paired_matrix = base_path / pathlib.Path(curr_base_path.name + "_output_paired.matrix")

        logger.info(f"Output path : {output_folder}")

        # =============================
        # img_type = configuration.SUPPORTED_IMAGE_TYPE.PNG
        config_launcher = Configuration_launcher(source_pictures_dir=source_pictures_dir.resolve(),
                                                 output_folder=output_folder.resolve(),
                                                 ground_truth_json=ground_truth_json.resolve(),
                                                 img_type=img_type,
                                                 overwrite_folder=args.overwrite,
                                                 args=args)
        try:
            # For profiling : cProfile.run("
            config_launcher.auto_launch()
            # ")
        except Exception as e:
            logger.error(f"Launch of whole configuration launcher aborted due to : {e}")
            logger.error(traceback.print_tb(e.__traceback__))


        # Create overview for simple results
        try:
            if args.tldr: Configuration_launcher.create_tldr(folder=output_folder, output_file=output_overview_file)
        except Exception as e:
            logger.error(f"Creation of TLDR aborted due to : {e}")
            logger.error(traceback.print_tb(e.__traceback__))

        try:
            if args.tldr_latex: Configuration_launcher.create_latex_tldr(folder=output_folder,output_file=output_latex_overview_file)
        except Exception as e:
            logger.error(f"Creation of TLDR LATEX aborted due to : {e}")
            logger.error(traceback.print_tb(e.__traceback__))

        # Create overview for paired results
        try:
            if args.pair_results: Configuration_launcher.create_paired_results(input_folder=output_folder,target_pair_folder=paired_output_folder,ground_truth_json=ground_truth_json)
        except Exception as e:
            logger.error(f"Creation of paired results aborted due to : {e}")
            logger.error(traceback.print_tb(e.__traceback__))

        try:
            if args.tldr_pairs: Configuration_launcher.create_tldr(folder=paired_output_folder,output_file=output_overview_paired_file)
        except Exception as e:
            logger.error(f"Creation of TLDR of paired results aborted due to : {e}")
            logger.error(traceback.print_tb(e.__traceback__))

        # Create matrixes
        try:
            if args.inclusion_matrix: Configuration_launcher.create_and_export_inclusion_matrix(folder=output_folder,output_file=output_similarity_matrix)
        except Exception as e:
            logger.error(f"Creation of inclusion matrix aborted due to : {e}")
            logger.error(traceback.print_tb(e.__traceback__))

        try:
            if args.inclusion_matrix_pairs: Configuration_launcher.create_and_export_pair_matrix(input_folder=output_folder,ground_truth_json=ground_truth_json,output_file=output_paired_matrix)
        except Exception as e:
            logger.error(f"Creation of quality paired matrix aborted due to : {e}")
            logger.error(traceback.print_tb(e.__traceback__))
