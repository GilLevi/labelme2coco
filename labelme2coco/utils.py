import json
import os
from pathlib import Path
import numpy as np
import jsonschema

image_schema = {
    "type": "object",
    "properties": {"file_name": {"type": "string"}, "id": {"type": "integer"}},
    "required": ["file_name", "id"],
}

segmentation_schema = {
    "type": "array",
    "items": {
        "type": "array",
        "items": {
            "type": "number",
        },
        "additionalItems": False,
    },
    "additionalItems": False,
}

annotation_schema = {
    "type": "object",
    "properties": {
        "image_id": {"type": "integer"},
        "category_id": {"type": "integer"},
        "segmentation": segmentation_schema,
    },
    "required": ["image_id", "category_id", "segmentation"],
}

category_schema = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
    "required": ["name", "id"],
}

coco_schema = {
    "type": "object",
    "properties": {
        "images": {"type": "array", "items": image_schema, "additionalItems": False},
        "annotations": {
            "type": "array",
            "items": annotation_schema,
            "additionalItems": False,
        },
        "categories": {
            "type": "array",
            "items": category_schema,
            "additionalItems": False,
        },
    },
    "required": ["images", "annotations", "categories"],
}


def read_and_validate_coco_annotation(coco_annotation_path: str) -> (dict, bool):
    """
    Reads coco formatted annotation file and validates its fields.
    """
    try:
        with open(coco_annotation_path) as json_file:
            coco_dict = json.load(json_file)
        jsonschema.validate(coco_dict, coco_schema)
        response = True
    except jsonschema.exceptions.ValidationError as e:
        print("well-formed but invalid JSON:", e)
        response = False
    except json.decoder.JSONDecodeError as e:
        print("poorly-formed text, not JSON:", e)
        response = False

    return coco_dict, response


def save_json(data, save_path):
    """
    Saves json formatted data (given as "data") as save_path
    Example inputs:
        data: {"image_id": 5}
        save_path: "dirname/coco.json"
    """
    # create dir if not present
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # export as json
    with open(save_path, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, separators=(",", ":"), cls=NumpyEncoder)


# type check when save json files
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def load_json(load_path: str, encoding: str = "utf-8"):
    """
    Loads json formatted data (given as "data") from load_path
    Encoding type can be specified with 'encoding' argument

    Example inputs:
        load_path: "dirname/coco.json"
    """
    # read from path
    with open(load_path, encoding=encoding) as json_file:
        data = json.load(json_file)
    return data


def list_files_recursively(directory: str, contains: list = [".json"], verbose: str = True) -> (list, list):
    """
    Walk given directory recursively and return a list of file path with desired extension

    Arguments
    -------
        directory : str
            "data/coco/"
        contains : list
            A list of strings to check if the target file contains them, example: ["coco.png", ".jpg", "jpeg"]
        verbose : bool
            If true, prints some results
    Returns
    -------
        relative_filepath_list : list
            List of file paths relative to given directory
        abs_filepath_list : list
            List of absolute file paths
    """

    # define verboseprint
    verboseprint = print if verbose else lambda *a, **k: None

    # walk directories recursively and find json files
    abs_filepath_list = []
    relative_filepath_list = []

    # r=root, d=directories, f=files
    for r, _, f in os.walk(directory):
        for file in f:
            # check if filename contains any of the terms given in contains list
            if any(strtocheck in file for strtocheck in contains):
                abs_filepath = os.path.join(r, file)
                abs_filepath_list.append(abs_filepath)
                relative_filepath = abs_filepath.split(directory)[-1]
                relative_filepath_list.append(relative_filepath)

    number_of_files = len(relative_filepath_list)
    folder_name = directory.split(os.sep)[-1]

    verboseprint("There are {} listed files in folder {}.".format(number_of_files, folder_name))

    return relative_filepath_list, abs_filepath_list
