# STD LIBRARIES
import json
import pathlib
import random
from PIL import Image, ImageDraw
from .picture_class import Picture
import cv2
import configuration
import logging
import pickle
import pprint
import ast

# PERSONAL LIBRARIES
TOP_K_EDGE = 1

class Custom_JSON_Encoder(json.JSONEncoder):
    '''
    Custom JSON Encoder to store Enum and custom configuration objects (for example) of the framework
    '''

    def default(self, o):
        if isinstance(o, pathlib.Path):
            return str(o)
        if isinstance(o, configuration.JSON_parsable_Enum):
            # If this is an object flagger as String equivalent, parse it as a String
            return str(o)
        if isinstance(o, configuration.JSON_parsable_Dict):
            # If this is an object flagged as dict equivalent, parse it as a dict.
            return o.__dict__
        return json.JSONEncoder.default(self, o)


class File_System():
    def __init__(self, conf: configuration.Default_configuration):
        self.conf = conf
        self.logger =  logging.getLogger('__main__.' + __name__)

        if conf.IMG_TYPE == configuration.SUPPORTED_IMAGE_TYPE.PNG:
            self.type = ".png"
        elif conf.IMG_TYPE == configuration.SUPPORTED_IMAGE_TYPE.BMP:
            self.type = ".bmp"
        else:
            raise Exception("IMG_TYPE not recognized as a valid type" + str(type))

    # ==== Disk read ====
    def safe_path(self, path):
        if type(path) is pathlib.PosixPath:
            logging.debug("Path specified is already a pathlib Path. ")
        elif type(path) is str:
            logging.debug("Path specified is a string.")
            path = pathlib.Path(path)
        else:
            raise Exception("Unknown path type")
        return path

    def fixed_choice(self):
        target_picture_path = self.safe_path('../../datasets/raw_phishing/comunidadejesusteama.org.br' + self.type)

        return target_picture_path

    def random_choice(self, target_dir):
        target_dir = self.safe_path(target_dir)

        pathlist = target_dir.glob('**/*' + self.type)
        target_picture_path = random.choice(list(pathlist))

        return target_picture_path

    def get_Pictures_from_directory(self, directory_path, class_name=Picture):
        directory_path = self.safe_path(directory_path)

        pathlist = directory_path.glob('**/*' + self.type)
        picture_list = []

        for i, path in enumerate(pathlist):
            tmp_Picture = class_name(id=i, conf=self.conf, path=path)

            # Store hash
            picture_list.append(tmp_Picture)

        return picture_list

    # ==== Disk write ====

    def clean_folder(self, target_dir):
        '''
        Remove 0-bytes size files
        :param target_dir:
        :return:
        '''

        pathlist = pathlib.Path(target_dir).glob('**/*' + self.type)
        for i, path in enumerate(pathlist):

            if path.stat().st_size == 0:
                path.unlink()
            curr_pic = cv2.imread(str(path))

            if curr_pic is None or curr_pic.shape == [] or curr_pic.shape[0] == 0 or curr_pic.shape[1] == 0:
                logging.error(f"Void picture (to delete ?) : {path}")
                # path.unlink()


    @staticmethod
    def save_json(obj, file_path : pathlib.Path):
        # TODO : To fix json_data = ast.literal_eval(json_data) ?  See : https://stackoverflow.com/questions/25707558/json-valueerror-expecting-property-name-line-1-column-2-char-1
        logger = logging.getLogger()

        # Create parents if they does not exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, cls=Custom_JSON_Encoder)

        logger.debug(f"File saved as {file_path}.")

    @staticmethod
    def load_json(file_path: pathlib.Path ):
        logger = logging.getLogger()
        data = None

        # !! : json.load() is for loading a file. json.loads() works with strings.
        # json.loads will load a json string into a python dict, json.dumps will dump a python dict to a json string,

        if file_path.is_file():
            # We have a valid file to load
            with open(str(file_path.resolve())) as json_file:
                data = json.load(json_file)
            logger.debug(f"File loaded from {file_path}.")

        else:
            raise Exception(f"Cannot load the provided path to json : path is not a valid file {file_path}")

        return data


## ============================== -------------- ==============================
#                                   ARCHIVES

'''
            with open(str(file_path.resolve())) as json_file:
                try :
                    data = json.loads(json_file)
                except Exception as e :
                    read_data_tmp = str(json_file.read())
                    try :
                        logger.error("JSON tried to load is not a correctly formatted json. Try to perform recover ...")
                        tmp_json_file = read_data_tmp.replace("'", '"')
                        data = json.loads(tmp_json_file)
                    except Exception as e :
                        logger.error("Recover failed. Loading file as text only ...")
                        data = read_data_tmp

'''

'''


    def write_configuration_to_folder(self, conf: configuration.Default_configuration):
        fn = "conf.txt"
        filepath = conf.OUTPUT_DIR / fn
        with filepath.open("w", encoding="utf-8") as f:
            f.write(pprint.pformat(vars(conf)))

        # data = json.dumps(results)
        # with filepath.open("w", encoding="utf-8") as f:
        #       f.write(data)

        self.logger.debug(f"Configuration file saved as {filepath}.")

    @staticmethod
    def save_json(obj, file_path : pathlib.Path):
        # Create path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Store object
        # f = open(str(tmp_file_path.resolve()), "w+")  # Overwrite and create if does not exist
        with open(str(file_path.resolve()), 'w+') as f:
            json.dump(obj, f, default=lambda x: x.__dict__)
        # Please see : https://stackoverflow.com/questions/10252010/serializing-class-instance-to-json
    '''


'''
try :
    with file_path.open("w", encoding="utf-8") as f:
        f.write(pprint.pformat(vars(obj)))
except Exception as e:
    try:
        logger.error(f"Saving Object to JSON file failed. Trying to save it as json dump... {file_path}")
        with open(str(file_path.resolve()), 'w+') as f:
            json.dump(obj, f, default=lambda x: x.__dict__)
    except Exception as e:
        logger.error(f"Saving Object to JSON file failed again. Abort saving. {file_path}")

'''

'''
UNUSED

@staticmethod
def save_obj(obj, file_path : pathlib.Path):
    # Create path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Store object
    with open(str(file_path.resolve()), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)# pickle.)

@staticmethod
def load_obj(file_path: pathlib.Path ):
    with open(str(file_path.resolve()), 'rb') as f:
        return pickle.load(f)
'''