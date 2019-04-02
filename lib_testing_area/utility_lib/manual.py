# ============================================= ------------------------------------ =============================================
#                                                           MANUAL LABOR

DEFAULT_TARGET_DIR = "../../datasets/raw_phishing_bmp/"
DEFAULT_BASELINE_PATH = "../../datasets/raw_phishing.json"
BMP_TARGET_DIR = "../../datasets/raw_phishing_bmp/"
BMP_BASELINE_PATH = "../../datasets/raw_phishing_bmp.json"

# Handle trucated image too
from PIL import Image, ImageFile
import pathlib
import json_class

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
    json_handler = json_class.JSON_GRAPHE()
    png_json = json_handler.import_json(pathlib.Path(DEFAULT_BASELINE_PATH)) # PNG one
    bmp_json = json_handler.replace_type(png_json, ".bmp")
    json_handler.json_to_export = bmp_json
    json_handler.json_export(str(BMP_BASELINE_PATH))

if __name__ == '__main__':
    create_bmp_from_png()