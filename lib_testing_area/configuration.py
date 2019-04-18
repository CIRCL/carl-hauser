# ==================== ------------------------ ====================
#                      Configuration declaration
from enum import Enum, auto

class SUPPORTED_IMAGE_TYPE(Enum):
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

# Threshold finder
class THRESHOLD_MODE(Enum):
    MIN_WRONG = auto()
    MEDIAN_WRONG = auto()
    MAX_WRONG = auto()
    MAXIMIZE_TRUE_POSITIVE = auto()

class Default_configuration():
    def __init__(self):
        # Inputs
        self.SOURCE_DIR = None
        self.GROUND_TRUTH_PATH = None
        self.IMG_TYPE = SUPPORTED_IMAGE_TYPE.PNG
        # Processing
        self.ALGO = ALGO_TYPE.A_HASH
        self.SELECTION_THREESHOLD = None #TODO : To fix and to use, to prevent "forced linked" if none
        #Threshold
        self.THREESHOLD_EVALUATION = THRESHOLD_MODE.MAXIMIZE_TRUE_POSITIVE
        # Output
        self.SAVE_PICTURE = False
        self.OUTPUT_DIR = None

# ==================== ------------------------ ====================
#                      ORB POSSIBLE CONFIGURATIONS
# See options there : https://docs.opencv.org/trunk/dc/d8c/namespacecvflann.html

class DISTANCE_TYPE(Enum):
    LEN_MIN = auto()
    LEN_MAX = auto()
    # LEN_MEAN = auto() # DOESNT WORK AT ALL
    MEAN_DIST_PER_PAIR = auto()
    MEAN_AND_MAX = auto()

class FILTER_TYPE(Enum):
    RATIO_BAD = auto() # NOT with KNN # DOESNT WORK WELL
    RATIO_CORRECT = auto() # ONLY with KNN
    FAR_THREESHOLD = auto() # NOT with KNN = THREESHOLD DISTANCE # DOESNT WORK WELL
    #### BASIC_THRESHOLD = auto() # DOESNT WORK WELL
    NO_FILTER = auto()
    RANSAC = auto()

class MATCH_TYPE(Enum):
    STD = auto() # Standard
    KNN = auto()

class DATASTRUCT_TYPE(Enum):
    BRUTE_FORCE = auto()
    # FLANN_KDTREE = auto()  # DOESNT WORK AT ALL
    FLANN_LSH = auto()

class CROSSCHECK(Enum):
    ENABLED = auto()
    DISABLED = auto()
    AUTO = auto()

class ORB_default_configuration(Default_configuration):
    def __init__(self):
        super().__init__()

        self.ORB_KEYPOINTS_NB = 500

        self.DISTANCE = DISTANCE_TYPE.LEN_MAX
        self.FILTER = FILTER_TYPE.NO_FILTER
        self.MATCH = MATCH_TYPE.STD
        self.DATASTRUCT = DATASTRUCT_TYPE.BRUTE_FORCE

        # Facultative depending on upper
        self.MATCH_K_FOR_KNN = 2

        self.FLANN_KDTREE_INDEX = 0
        self.FLANN_KDTREE_INDEX_params = dict(algorithm=self.FLANN_KDTREE_INDEX, trees=5)
        self.FLANN_KDTREE_SEARCH_params = dict(checks=50)

        self.FLANN_LSH_INDEX = 6
        self.FLANN_LSH_INDEX_params = dict(algorithm=self.FLANN_LSH_INDEX, table_number=6, key_size=12, multi_probe_level=1)
        self.FLANN_LSH_SEARCH_params = dict(checks=50)  # or pass empty dictionary
        self.FLANN_LSH_INDEX_params_light = dict(algorithm=self.FLANN_LSH_INDEX, table_number=6)

        # Crosscheck is handled automatically
        self.CROSSCHECK = CROSSCHECK.AUTO

# ==================== ------------------------ ====================
#                      BoW ORB POSSIBLE CONFIGURATIONS
#

class BOW_CMP_HIST(Enum):
    CORREL = auto() # Standard
    BHATTACHARYYA = auto()

class BoW_ORB_default_configuration(Default_configuration):
    def __init__(self):
        super().__init__()

        self.ORB_KEYPOINTS_NB = 500

        # BOW SPECIFIC
        self.BOW_SIZE = 100
        self.BOW_CMP_HIST = BOW_CMP_HIST.CORREL


# ==================== ------------------------ ====================
#                        Custom configuration