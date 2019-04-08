# ==================== ------------------------ ====================
#                       Results declaration
from enum import Enum, auto

class RESULTS():
    def __init__(self):
        self.TOTAL_TIME = None

        self.TIME_TO_LOAD_PICTURES = None

        self.TIME_TOTAL_PRE_COMPUTING = None
        self.TIME_REQUEST_PICTURE_COMPUTING = None
        self.TIME_PER_PICTURE_PRE_COMPUTING = None

        self.TIME_TOTAL_MATCHING = None
        self.TIME_LIST_MATCHING = None
        self.TIME_PER_PICTURE_MATCHING = None

        self.NB_PICTURE = None
        self.TRUE_POSITIVE_RATE = None
