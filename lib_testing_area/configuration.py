# ==================== ------------------------ ====================
#                      Configuration declaration
from enum import Enum, auto
from OpenCV.opencv import DISTANCE_TYPE as ORB_DISTANCE, FILTER_TYPE as ORB_FILTER_TYPE, MATCH_TYPE as ORB_MATCH_TYPE, DATASTRUCT_TYPE as ORB_DATASTRUCT_TYPE

class ALGO_TYPE(Enum):
    A_HASH = auto()
    P_HASH = auto()
    P_HASH_SIMPLE = auto()
    D_HASH = auto()
    D_HASH_VERTICAL = auto()
    W_HASH = auto()
    TLSH = auto()
    TLSH_NO_LENGTH = auto()
    ORB = auto()

class Default_configuration():
    ALGO = ALGO_TYPE.A_HASH
    SAVE_PICTURE = False

class ORB_default_configuration(Default_configuration):
    DISTANCE = ORB_DISTANCE.LEN_MAX
    FILTER = ORB_FILTER_TYPE.NO_FILTER
    MATCH = ORB_MATCH_TYPE.STD
    DATASTRUCT = ORB_DATASTRUCT_TYPE.BF

class Supported_image_type():
    PNG = auto()
    BMP = auto()

# ==================== ------------------------ ====================
#                        Custom configuration