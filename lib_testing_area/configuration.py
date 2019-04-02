# ==================== ------------------------ ====================
#                      Configuration declaration
from enum import Enum, auto

class SUPPORTED_IMAGE_TYPE():
    PNG = auto()
    BMP = auto()

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
    # Inputs
    SOURCE_DIR = None
    GROUND_TRUTH_PATH = None
    IMG_TYPE = SUPPORTED_IMAGE_TYPE.PNG
    # Processing
    ALGO = ALGO_TYPE.A_HASH
    SELECTION_THREESHOLD = None #TODO : To fix and to use, to prevent "forced linked" if none
    # Output
    SAVE_PICTURE = False
    OUTPUT_DIR = None

# ==================== ------------------------ ====================
#                      ORB POSSIBLE CONFIGURATIONS
# See options there : https://docs.opencv.org/trunk/dc/d8c/namespacecvflann.html

class DISTANCE_TYPE(Enum):
    LEN_MIN = auto()
    LEN_MAX = auto()
    LEN_MEAN = auto()
    MEAN_DIST_PER_PAIR = auto()
    MEAN_AND_MAX = auto()

class FILTER_TYPE(Enum):
    RATIO_BAD = auto() # NOT with KNN
    RATIO_CORRECT = auto() # ONLY with KNN
    FAR_THREESHOLD = auto() # NOT with KNN = THREESHOLD DISTANCE
    NO_FILTER = auto()

class MATCH_TYPE(Enum):
    STD = auto() # Standard
    KNN = auto()

class DATASTRUCT_TYPE(Enum):
    BRUTE_FORCE = auto()
    FLANN_KDTREE = auto()
    FLANN_LSH = auto()

class CROSSCHECK(Enum):
    ENABLED = auto()
    DISABLED = auto()
    AUTO = auto()

class ORB_default_configuration(Default_configuration):
    ORB_KEYPOINTS_NB = 500

    DISTANCE = DISTANCE_TYPE.LEN_MAX
    FILTER = FILTER_TYPE.NO_FILTER
    MATCH = MATCH_TYPE.STD
    DATASTRUCT = DATASTRUCT_TYPE.BRUTE_FORCE

    # Facultative depending on upper
    MATCH_K_FOR_KNN = 2

    FLANN_KDTREE_INDEX = 0
    FLANN_KDTREE_INDEX_params = dict(algorithm=FLANN_KDTREE_INDEX, trees=5)
    FLANN_KDTREE_SEARCH_params = dict(checks=50)

    FLANN_LSH_INDEX = 6
    FLANN_LSH_INDEX_params = dict(algorithm=FLANN_LSH_INDEX, table_number=6, key_size=12, multi_probe_level=1)
    FLANN_LSH_SEARCH_params = dict(checks=50)  # or pass empty dictionary
    FLANN_LSH_INDEX_params_light = dict(algorithm=FLANN_LSH_INDEX, table_number=6)

    # Crosscheck is handled automatically
    CROSSCHECK = CROSSCHECK.AUTO

# ==================== ------------------------ ====================
#                        Custom configuration