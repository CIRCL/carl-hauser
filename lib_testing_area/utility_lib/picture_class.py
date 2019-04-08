import pathlib
from PIL import Image
import configuration

class Picture():
    def __init__(self, id, conf: configuration.Default_configuration, shape: str = "image", path: pathlib.PosixPath = None):
        self.id = id
        self.conf = conf

        self.shape = shape
        self.path = path
        self.matched = False
        self.sorted_matching_picture_list = []

        # Hashing related attributes
        self.hash = None
        self.distance = None

        # Descriptors related attributes
        self.key_points = None
        self.description = None
        self.image = self.load_image(self.path)

        # Multipurpose storage, e.g. store some useful class for processing.
        # self.storage = None
        self.matches = None

    def load_image(self, path: pathlib.PosixPath):
        if path is None or path == "":
            return None

        return Image.open(str(path))

    def is_same_picture_as(self, pic1):
        # TODO : Except on SHA1 hash ?
        return self.path == pic1.path

    def to_node_json_object(self):
        tmp_obj = {}
        tmp_obj["id"] = self.id
        tmp_obj["shape"] = self.shape
        tmp_obj["image"] = self.path.name
        return tmp_obj
