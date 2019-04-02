# ==================== ------------------------ ====================
#                       Configuration launcher
# STD imports
import logging
import pathlib

# Own imports
import utility_lib.filesystem_lib as filesystem_lib
import configuration

class Configuration_launcher():
    def __init__(self, target_pictures_dir: pathlib.Path, ground_truth_json : pathlib.Path, img_type : configuration.Supported_image_type):
        self.logger = logging.getLogger(__name__)
        self.target_pictures_dir = target_pictures_dir
        self.ground_truth_json = ground_truth_json
        self.img_type = img_type

        self.logger.warning("Creation of filesystem handler : deletion of 0-sized pictures.")
        self.filesystem_handler = filesystem_lib.File_System(type=img_type)
        self.filesystem_handler.clean_folder(target_dir)

    def auto_launch(self):
        self.logger.info("==== ----- LAUNCHING AUTO CONF LAUNCHER ---- ==== ")
        self.auto_launch_image_hash()
        self.auto_launch_tlsh()
        self.auto_launch_orb()

    def auto_launch_image_hash(self):
        self.logger.info("==== ----- LAUNCHING IMAGE HASH ALGOS ---- ==== ")

        # Create conf
        # Launch
        # redo
        eh = Image_hash_execution_handler(target_dir=target_dir, Local_Picture=Local_Picture)
        eh.do_full_test()

    def auto_launch_tlsh(self):
        self.logger.info("==== ----- LAUNCHING TLSH algos ---- ==== ")

        # Create conf
        # Launch
        # redo
        eh = TLSH_execution_handler(target_dir=target_dir, Local_Picture=Local_Picture)
        eh.do_full_test()

    def auto_launch_orb(self):
        self.logger.info("==== ----- LAUNCHING ORB algos ---- ==== ")

        # Create conf
        # Launch
        # redo
        eh = OpenCV_execution_handler(target_dir=target_dir, Local_Picture=Local_Picture, save_picture=False)
        eh.storage = Matcher()
        eh.printer = Custom_printer()
        eh.do_full_test()

if __name__ == '__main__':
    target_dir = pathlib.Path("../../datasets/raw_phishing/")
    ground_truth_json = pathlib.Path("../../datasets/raw_phishing.json")
    img_type = configuration.Supported_image_type.PNG

    config_launcher = Configuration_launcher(target_pictures_dir= target_dir, ground_truth_json=ground_truth_json, img_type=img_type)
    config_launcher.auto_launch()