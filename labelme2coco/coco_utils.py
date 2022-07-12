import copy
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import numpy as np
from tqdm import tqdm
from labelme2coco.shapely_utils import ShapelyAnnotation, box, get_shapely_multipolygon


class Coco:
    def __init__(
        self,
        name=None,
        image_dir=None,
        remapping_dict=None,
        ignore_negative_samples=False,
        clip_bboxes_to_img_dims=False,
        image_id_setting="auto",
    ):
        """
        Creates Coco object.

        Args:
            name: str
                Name of the Coco dataset, it determines exported json name.
            image_dir: str
                Base file directory that contains dataset images. Required for dataset merging.
            remapping_dict: dict
                {1:0, 2:1} maps category id 1 to 0 and category id 2 to 1
            ignore_negative_samples: bool
                If True ignores images without annotations in all operations.
            image_id_setting: str
                how to assign image ids while exporting can be
                    auto --> will assign id from scratch (<CocoImage>.id will be ignored)
                    manual --> you will need to provide image ids in <CocoImage> instances (<CocoImage>.id can not be None)
        """
        if image_id_setting not in ["auto", "manual"]:
            raise ValueError("image_id_setting must be either 'auto' or 'manual'")
        self.name = name
        self.image_dir = image_dir
        self.remapping_dict = remapping_dict
        self.ignore_negative_samples = ignore_negative_samples
        self.categories = []
        self.images = []
        self._stats = None
        self.clip_bboxes_to_img_dims = clip_bboxes_to_img_dims
        self.image_id_setting = image_id_setting

    def add_categories_from_coco_category_list(self, coco_category_list):
        """
        Creates CocoCategory object using coco category list.

        Args:
            coco_category_list: List[Dict]
                [
                    {"supercategory": "person", "id": 1, "name": "person"},
                    {"supercategory": "vehicle", "id": 2, "name": "bicycle"}
                ]
        """

        for coco_category in coco_category_list:
            if self.remapping_dict is not None:
                for source_id in self.remapping_dict.keys():
                    if coco_category["id"] == source_id:
                        target_id = self.remapping_dict[source_id]
                        coco_category["id"] = target_id

            self.add_category(CocoCategory.from_coco_category(coco_category))

    def add_category(self, category):
        """
        Adds category to this Coco instance

        Args:
            category: CocoCategory
        """

        # assert type(category) == CocoCategory, "category must be a CocoCategory instance"
        if not isinstance(category, CocoCategory):
            raise TypeError("category must be a CocoCategory instance")
        self.categories.append(category)

    def add_image(self, image):
        """
        Adds image to this Coco instance

        Args:
            image: CocoImage
        """

        if self.image_id_setting == "manual" and image.id is None:
            raise ValueError("image id should be manually set for image_id_setting='manual'")
        self.images.append(image)

    def update_categories(self, desired_name2id, update_image_filenames=False):
        """
        Rearranges category mapping of given COCO object based on given desired_name2id.
        Can also be used to filter some of the categories.

        Args:
            desired_name2id: dict
                {"big_vehicle": 1, "car": 2, "human": 3}
            update_image_filenames: bool
                If True, updates coco image file_names with absolute file paths.
        """
        # init vars
        currentid2desiredid_mapping = {}
        updated_coco = Coco(
            name=self.name,
            image_dir=self.image_dir,
            remapping_dict=self.remapping_dict,
            ignore_negative_samples=self.ignore_negative_samples,
        )
        # create category id mapping (currentid2desiredid_mapping)
        for coco_category in copy.deepcopy(self.categories):
            current_category_id = coco_category.id
            current_category_name = coco_category.name
            if current_category_name in desired_name2id.keys():
                currentid2desiredid_mapping[current_category_id] = desired_name2id[current_category_name]
            else:
                # ignore categories that are not included in desired_name2id
                currentid2desiredid_mapping[current_category_id] = None

        # add updated categories
        for name in desired_name2id.keys():
            updated_coco_category = CocoCategory(id=desired_name2id[name], name=name, supercategory=name)
            updated_coco.add_category(updated_coco_category)

        # add updated images & annotations
        for coco_image in copy.deepcopy(self.images):
            updated_coco_image = CocoImage.from_coco_image_dict(coco_image.json)
            # update filename to abspath
            file_name_is_abspath = True if os.path.abspath(coco_image.file_name) == coco_image.file_name else False
            if update_image_filenames and not file_name_is_abspath:
                updated_coco_image.file_name = str(Path(os.path.abspath(self.image_dir)) / coco_image.file_name)
            # update annotations
            for coco_annotation in coco_image.annotations:
                current_category_id = coco_annotation.category_id
                desired_category_id = currentid2desiredid_mapping[current_category_id]
                # append annotations with category id present in desired_name2id
                if desired_category_id is not None:
                    # update cetegory id
                    coco_annotation.category_id = desired_category_id
                    # append updated annotation to target coco dict
                    updated_coco_image.add_annotation(coco_annotation)
            updated_coco.add_image(updated_coco_image)

        # overwrite instance
        self.__class__ = updated_coco.__class__
        self.__dict__ = updated_coco.__dict__

    def merge(self, coco, desired_name2id=None, verbose=1):
        """
        Combines the images/annotations/categories of given coco object with current one.

        Args:
            coco : sahi.utils.coco.Coco instance
                A COCO dataset object
            desired_name2id : dict
                {"human": 1, "car": 2, "big_vehicle": 3}
            verbose: bool
                If True, merging info is printed
        """
        if self.image_dir is None or coco.image_dir is None:
            raise ValueError("image_dir should be provided for merging.")
        if verbose:
            if not desired_name2id:
                print("'desired_name2id' is not specified, combining all categories.")

        # create desired_name2id by combining all categories, if desired_name2id is not specified
        coco1 = self
        coco2 = coco
        category_ind = 0
        if desired_name2id is None:
            desired_name2id = {}
            for coco in [coco1, coco2]:
                temp_categories = copy.deepcopy(coco.json_categories)
                for temp_category in temp_categories:
                    if temp_category["name"] not in desired_name2id:
                        desired_name2id[temp_category["name"]] = category_ind
                        category_ind += 1
                    else:
                        continue

        # update categories and image paths
        for coco in [coco1, coco2]:
            coco.update_categories(desired_name2id=desired_name2id, update_image_filenames=True)

        # combine images and categories
        coco1.images.extend(coco2.images)
        self.images: List[CocoImage] = coco1.images
        self.categories = coco1.categories

        # print categories
        if verbose:
            print(
                "Categories are formed as:\n",
                self.json_categories,
            )

    @classmethod
    def from_coco_dict_or_path(
        cls,
        coco_dict_or_path: Union[Dict, str],
        image_dir: Optional[str] = None,
        remapping_dict: Optional[Dict] = None,
        ignore_negative_samples: bool = False,
        clip_bboxes_to_img_dims: bool = False,
    ):
        """
        Creates coco object from COCO formatted dict or COCO dataset file path.

        Args:
            coco_dict_or_path: dict/str or List[dict/str]
                COCO formatted dict or COCO dataset file path
                List of COCO formatted dict or COCO dataset file path
            image_dir: str
                Base file directory that contains dataset images. Required for merging and yolov5 conversion.
            remapping_dict: dict
                {1:0, 2:1} maps category id 1 to 0 and category id 2 to 1
            ignore_negative_samples: bool
                If True ignores images without annotations in all operations.
            clip_bboxes_to_img_dims: bool = False
                Limits bounding boxes to image dimensions.

        Properties:
            images: list of CocoImage
            category_mapping: dict
        """
        # init coco object
        coco = cls(
            image_dir=image_dir,
            remapping_dict=remapping_dict,
            ignore_negative_samples=ignore_negative_samples,
            clip_bboxes_to_img_dims=clip_bboxes_to_img_dims,
        )

        if type(coco_dict_or_path) not in [str, dict]:
            raise TypeError("coco_dict_or_path should be a dict or str")

        # load coco dict if path is given
        if type(coco_dict_or_path) == str:
            coco_dict = load_json(coco_dict_or_path)
        else:
            coco_dict = coco_dict_or_path

        # arrange image id to annotation id mapping
        coco.add_categories_from_coco_category_list(coco_dict["categories"])
        image_id_to_annotation_list = get_imageid2annotationlist_mapping(coco_dict)
        category_mapping = coco.category_mapping

        # https://github.com/obss/sahi/issues/98
        image_id_set: Set = set()

        for coco_image_dict in tqdm(coco_dict["images"], "Loading coco annotations"):
            coco_image = CocoImage.from_coco_image_dict(coco_image_dict)
            image_id = coco_image_dict["id"]
            # https://github.com/obss/sahi/issues/98
            if image_id in image_id_set:
                print(f"duplicate image_id: {image_id}, will be ignored.")
                continue
            else:
                image_id_set.add(image_id)
            # select annotations of the image
            annotation_list = image_id_to_annotation_list[image_id]
            for coco_annotation_dict in annotation_list:
                # apply category remapping if remapping_dict is provided
                if coco.remapping_dict is not None:
                    # apply category remapping (id:id)
                    category_id = coco.remapping_dict[coco_annotation_dict["category_id"]]
                    # update category id
                    coco_annotation_dict["category_id"] = category_id
                else:
                    category_id = coco_annotation_dict["category_id"]
                # get category name (id:name)
                category_name = category_mapping[category_id]
                coco_annotation = CocoAnnotation.from_coco_annotation_dict(
                    category_name=category_name, annotation_dict=coco_annotation_dict
                )
                coco_image.add_annotation(coco_annotation)
            coco.add_image(coco_image)

        if clip_bboxes_to_img_dims:
            coco = coco.get_coco_with_clipped_bboxes()
        return coco

    @property
    def json_categories(self):
        categories = []
        for category in self.categories:
            categories.append(category.json)
        return categories

    @property
    def category_mapping(self):
        category_mapping = {}
        for category in self.categories:
            category_mapping[category.id] = category.name
        return category_mapping

    @property
    def json(self):
        return create_coco_dict(
            images=self.images,
            categories=self.json_categories,
            ignore_negative_samples=self.ignore_negative_samples,
            image_id_setting=self.image_id_setting,
        )

    @property
    def prediction_array(self):
        return create_coco_prediction_array(
            images=self.images,
            ignore_negative_samples=self.ignore_negative_samples,
            image_id_setting=self.image_id_setting,
        )

    @property
    def stats(self):
        if not self._stats:
            self.calculate_stats()
        return self._stats

    def calculate_stats(self):
        """
        Iterates over all annotations and calculates total number of
        """
        # init all stats
        num_annotations = 0
        num_images = len(self.images)
        num_negative_images = 0
        num_categories = len(self.json_categories)
        category_name_to_zero = {category["name"]: 0 for category in self.json_categories}
        category_name_to_inf = {category["name"]: float("inf") for category in self.json_categories}
        num_images_per_category = copy.deepcopy(category_name_to_zero)
        num_annotations_per_category = copy.deepcopy(category_name_to_zero)
        min_annotation_area_per_category = copy.deepcopy(category_name_to_inf)
        max_annotation_area_per_category = copy.deepcopy(category_name_to_zero)
        min_num_annotations_in_image = float("inf")
        max_num_annotations_in_image = 0
        total_annotation_area = 0
        min_annotation_area = 1e10
        max_annotation_area = 0
        for image in self.images:
            image_contains_category = {}
            for annotation in image.annotations:
                annotation_area = annotation.area
                total_annotation_area += annotation_area
                num_annotations_per_category[annotation.category_name] += 1
                image_contains_category[annotation.category_name] = 1
                # update min&max annotation area
                if annotation_area > max_annotation_area:
                    max_annotation_area = annotation_area
                if annotation_area < min_annotation_area:
                    min_annotation_area = annotation_area
                if annotation_area > max_annotation_area_per_category[annotation.category_name]:
                    max_annotation_area_per_category[annotation.category_name] = annotation_area
                if annotation_area < min_annotation_area_per_category[annotation.category_name]:
                    min_annotation_area_per_category[annotation.category_name] = annotation_area
            # update num_negative_images
            if len(image.annotations) == 0:
                num_negative_images += 1
            # update num_annotations
            num_annotations += len(image.annotations)
            # update num_images_per_category
            num_images_per_category = dict(Counter(num_images_per_category) + Counter(image_contains_category))
            # update min&max_num_annotations_in_image
            num_annotations_in_image = len(image.annotations)
            if num_annotations_in_image > max_num_annotations_in_image:
                max_num_annotations_in_image = num_annotations_in_image
            if num_annotations_in_image < min_num_annotations_in_image:
                min_num_annotations_in_image = num_annotations_in_image
        if (num_images - num_negative_images) > 0:
            avg_num_annotations_in_image = num_annotations / (num_images - num_negative_images)
            avg_annotation_area = total_annotation_area / num_annotations
        else:
            avg_num_annotations_in_image = 0
            avg_annotation_area = 0

        self._stats = {
            "num_images": num_images,
            "num_annotations": num_annotations,
            "num_categories": num_categories,
            "num_negative_images": num_negative_images,
            "num_images_per_category": num_images_per_category,
            "num_annotations_per_category": num_annotations_per_category,
            "min_num_annotations_in_image": min_num_annotations_in_image,
            "max_num_annotations_in_image": max_num_annotations_in_image,
            "avg_num_annotations_in_image": avg_num_annotations_in_image,
            "min_annotation_area": min_annotation_area,
            "max_annotation_area": max_annotation_area,
            "avg_annotation_area": avg_annotation_area,
            "min_annotation_area_per_category": min_annotation_area_per_category,
            "max_annotation_area_per_category": max_annotation_area_per_category,
        }

    def split_coco_as_train_val(self, train_split_rate=0.9, numpy_seed=0):
        """
        Split images into train-val and returns them as sahi.utils.coco.Coco objects.

        Args:
            train_split_rate: float
            numpy_seed: int
                To fix the numpy seed.

        Returns:
            result : dict
                {
                    "train_coco": "",
                    "val_coco": "",
                }
        """
        # fix numpy numpy seed
        np.random.seed(numpy_seed)

        # divide images
        num_images = len(self.images)
        shuffled_images = copy.deepcopy(self.images)
        np.random.shuffle(shuffled_images)
        num_train = int(num_images * train_split_rate)
        train_images = shuffled_images[:num_train]
        val_images = shuffled_images[num_train:]

        # form train val coco objects
        train_coco = Coco(
            name=self.name if self.name else "split" + "_train",
            image_dir=self.image_dir,
        )
        train_coco.images = train_images
        train_coco.categories = self.categories

        val_coco = Coco(name=self.name if self.name else "split" + "_val", image_dir=self.image_dir)
        val_coco.images = val_images
        val_coco.categories = self.categories

        # return result
        return {
            "train_coco": train_coco,
            "val_coco": val_coco,
        }

    def export_as_yolov5(self, output_dir, train_split_rate=1, numpy_seed=0, mp=False):
        """
        Exports current COCO dataset in ultralytics/yolov5 format.
        Creates train val folders with image symlinks and txt files and a data yaml file.

        Args:
            output_dir: str
                Export directory.
            train_split_rate: float
                If given 1, will be exported as train split.
                If given 0, will be exported as val split.
                If in between 0-1, both train/val splits will be calculated and exported.
            numpy_seed: int
                To fix the numpy seed.
            mp: bool
                If True, multiprocess mode is on.
                Should be called in 'if __name__ == __main__:' block.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                'Please run "pip install -U pyyaml" ' "to install yaml first for yolov5 formatted exporting."
            )

        # set split_mode
        if 0 < train_split_rate and train_split_rate < 1:
            split_mode = "TRAINVAL"
        elif train_split_rate == 0:
            split_mode = "VAL"
        elif train_split_rate == 1:
            split_mode = "TRAIN"
        else:
            raise ValueError("train_split_rate cannot be <0 or >1")

        # split dataset
        if split_mode == "TRAINVAL":
            result = self.split_coco_as_train_val(
                train_split_rate=train_split_rate,
                numpy_seed=numpy_seed,
            )
            train_coco = result["train_coco"]
            val_coco = result["val_coco"]
        elif split_mode == "TRAIN":
            train_coco = self
            val_coco = None
        elif split_mode == "VAL":
            train_coco = None
            val_coco = self

        # create train val image dirs
        train_dir = ""
        val_dir = ""
        if split_mode in ["TRAINVAL", "TRAIN"]:
            train_dir = Path(os.path.abspath(output_dir)) / "train/"
            train_dir.mkdir(parents=True, exist_ok=True)  # create dir
        if split_mode in ["TRAINVAL", "VAL"]:
            val_dir = Path(os.path.abspath(output_dir)) / "val/"
            val_dir.mkdir(parents=True, exist_ok=True)  # create dir

        # create image symlinks and annotation txts
        if split_mode in ["TRAINVAL", "TRAIN"]:
            export_yolov5_images_and_txts_from_coco_object(
                output_dir=train_dir,
                coco=train_coco,
                ignore_negative_samples=self.ignore_negative_samples,
                mp=mp,
            )
        if split_mode in ["TRAINVAL", "VAL"]:
            export_yolov5_images_and_txts_from_coco_object(
                output_dir=val_dir,
                coco=val_coco,
                ignore_negative_samples=self.ignore_negative_samples,
                mp=mp,
            )

        # create yolov5 data yaml
        data = {
            "train": str(train_dir),
            "val": str(val_dir),
            "nc": len(self.category_mapping),
            "names": list(self.category_mapping.values()),
        }
        yaml_path = str(Path(output_dir) / "data.yml")
        with open(yaml_path, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=None)

    def get_subsampled_coco(self, subsample_ratio: int = 2, category_id: int = None):
        """
        Subsamples images with subsample_ratio and returns as sahi.utils.coco.Coco object.

        Args:
            subsample_ratio: int
                10 means take every 10th image with its annotations
            category_id: int
                subsample only images containing given category_id, if -1 then subsamples negative samples
        Returns:
            subsampled_coco: sahi.utils.coco.Coco
        """
        subsampled_coco = Coco(
            name=self.name,
            image_dir=self.image_dir,
            remapping_dict=self.remapping_dict,
            ignore_negative_samples=self.ignore_negative_samples,
        )
        subsampled_coco.add_categories_from_coco_category_list(self.json_categories)

        if category_id is not None:
            # get images that contain given category id
            images_that_contain_category: List[CocoImage] = []
            for image in self.images:
                category_id_to_contains = defaultdict(lambda: 0)
                annotation: CocoAnnotation
                for annotation in image.annotations:
                    category_id_to_contains[annotation.category_id] = 1
                if category_id_to_contains[category_id]:
                    add_this_image = True
                elif category_id == -1 and len(image.annotations) == 0:
                    # if category_id is given as -1, select negative samples
                    add_this_image = True
                else:
                    add_this_image = False

                if add_this_image:
                    images_that_contain_category.append(image)

            # get images that does not contain given category id
            images_that_doesnt_contain_category: List[CocoImage] = []
            for image in self.images:
                category_id_to_contains = defaultdict(lambda: 0)
                annotation: CocoAnnotation
                for annotation in image.annotations:
                    category_id_to_contains[annotation.category_id] = 1
                if category_id_to_contains[category_id]:
                    add_this_image = False
                elif category_id == -1 and len(image.annotations) == 0:
                    # if category_id is given as -1, dont select negative samples
                    add_this_image = False
                else:
                    add_this_image = True

                if add_this_image:
                    images_that_doesnt_contain_category.append(image)

        if category_id:
            selected_images = images_that_contain_category
            # add images that does not contain given category without subsampling
            for image_ind in range(len(images_that_doesnt_contain_category)):
                subsampled_coco.add_image(images_that_doesnt_contain_category[image_ind])
        else:
            selected_images = self.images
        for image_ind in range(0, len(selected_images), subsample_ratio):
            subsampled_coco.add_image(selected_images[image_ind])

        return subsampled_coco

    def get_upsampled_coco(self, upsample_ratio: int = 2, category_id: int = None):
        """
        Upsamples images with upsample_ratio and returns as sahi.utils.coco.Coco object.

        Args:
            upsample_ratio: int
                10 means copy each sample 10 times
            category_id: int
                upsample only images containing given category_id, if -1 then upsamples negative samples
        Returns:
            upsampled_coco: sahi.utils.coco.Coco
        """
        upsampled_coco = Coco(
            name=self.name,
            image_dir=self.image_dir,
            remapping_dict=self.remapping_dict,
            ignore_negative_samples=self.ignore_negative_samples,
        )
        upsampled_coco.add_categories_from_coco_category_list(self.json_categories)
        for ind in range(upsample_ratio):
            for image_ind in range(len(self.images)):
                # calculate add_this_image
                if category_id is not None:
                    category_id_to_contains = defaultdict(lambda: 0)
                    annotation: CocoAnnotation
                    for annotation in self.images[image_ind].annotations:
                        category_id_to_contains[annotation.category_id] = 1
                    if category_id_to_contains[category_id]:
                        add_this_image = True
                    elif category_id == -1 and len(self.images[image_ind].annotations) == 0:
                        # if category_id is given as -1, select negative samples
                        add_this_image = True
                    elif ind == 0:
                        # in first iteration add all images
                        add_this_image = True
                    else:
                        add_this_image = False
                else:
                    add_this_image = True

                if add_this_image:
                    upsampled_coco.add_image(self.images[image_ind])

        return upsampled_coco

    def get_area_filtered_coco(self, min=0, max=float("inf"), intervals_per_category=None):
        """
        Filters annotation areas with given min and max values and returns remaining
        images as sahi.utils.coco.Coco object.

        Args:
            min: int
                minimum allowed area
            max: int
                maximum allowed area
            intervals_per_category: dict of dicts
                {
                    "human": {"min": 20, "max": 10000},
                    "vehicle": {"min": 50, "max": 15000},
                }
        Returns:
            area_filtered_coco: sahi.utils.coco.Coco
        """
        area_filtered_coco = Coco(
            name=self.name,
            image_dir=self.image_dir,
            remapping_dict=self.remapping_dict,
            ignore_negative_samples=self.ignore_negative_samples,
        )
        area_filtered_coco.add_categories_from_coco_category_list(self.json_categories)
        for image in self.images:
            is_valid_image = True
            for annotation in image.annotations:
                if intervals_per_category is not None and annotation.category_name in intervals_per_category.keys():
                    category_based_min = intervals_per_category[annotation.category_name]["min"]
                    category_based_max = intervals_per_category[annotation.category_name]["max"]
                    if annotation.area < category_based_min or annotation.area > category_based_max:
                        is_valid_image = False
                if annotation.area < min or annotation.area > max:
                    is_valid_image = False
            if is_valid_image:
                area_filtered_coco.add_image(image)

        return area_filtered_coco

    def get_coco_with_clipped_bboxes(self):
        """
        Limits overflowing bounding boxes to image dimensions.
        """
        from sahi.slicing import annotation_inside_slice

        coco = Coco(
            name=self.name,
            image_dir=self.image_dir,
            remapping_dict=self.remapping_dict,
            ignore_negative_samples=self.ignore_negative_samples,
        )
        coco.add_categories_from_coco_category_list(self.json_categories)

        for coco_img in self.images:
            img_dims = [0, 0, coco_img.width, coco_img.height]
            coco_image = CocoImage(
                file_name=coco_img.file_name, height=coco_img.height, width=coco_img.width, id=coco_img.id
            )
            for coco_ann in coco_img.annotations:
                ann_dict: Dict = coco_ann.json
                if annotation_inside_slice(annotation=ann_dict, slice_bbox=img_dims):
                    shapely_ann = coco_ann.get_sliced_coco_annotation(img_dims)
                    bbox = ShapelyAnnotation.to_coco_bbox(shapely_ann._shapely_annotation)
                    coco_ann_from_shapely = CocoAnnotation(
                        bbox=bbox,
                        category_id=coco_ann.category_id,
                        category_name=coco_ann.category_name,
                        image_id=coco_ann.image_id,
                    )
                    coco_image.add_annotation(coco_ann_from_shapely)
                else:
                    continue
            coco.add_image(coco_image)
        return coco


class CocoImage:
    @classmethod
    def from_coco_image_dict(cls, image_dict):
        """
        Creates CocoImage object from COCO formatted image dict (with fields "id", "file_name", "height" and "weight").

        Args:
            image_dict: dict
                COCO formatted image dict (with fields "id", "file_name", "height" and "weight")
        """
        return cls(
            id=image_dict["id"],
            file_name=image_dict["file_name"],
            height=image_dict["height"],
            width=image_dict["width"],
        )

    def __init__(self, file_name: str, height: int, width: int, id: int = None):
        """
        Creates CocoImage object

        Args:
            id : int
                Image id
            file_name : str
                Image path
            height : int
                Image height in pixels
            width : int
                Image width in pixels
        """
        self.id = int(id) if id else id
        self.file_name = file_name
        self.height = int(height)
        self.width = int(width)
        self.annotations = []  # list of CocoAnnotation that belong to this image
        self.predictions = []  # list of CocoPrediction that belong to this image

    def add_annotation(self, annotation):
        """
        Adds annotation to this CocoImage instance

        annotation : CocoAnnotation
        """

        if not isinstance(annotation, CocoAnnotation):
            raise TypeError("annotation must be a CocoAnnotation instance")
        self.annotations.append(annotation)

    def add_prediction(self, prediction):
        """
        Adds prediction to this CocoImage instance

        prediction : CocoPrediction
        """

        if not isinstance(prediction, CocoPrediction):
            raise TypeError("prediction must be a CocoPrediction instance")
        self.predictions.append(prediction)

    @property
    def json(self):
        return {
            "id": self.id,
            "file_name": self.file_name,
            "height": self.height,
            "width": self.width,
        }

    def __repr__(self):
        return f"""CocoImage<
    id: {self.id},
    file_name: {self.file_name},
    height: {self.height},
    width: {self.width},
    annotations: List[CocoAnnotation],
    predictions: List[CocoPrediction]>"""


class CocoCategory:
    """
    COCO formatted category.
    """

    def __init__(self, id=None, name=None, supercategory=None):
        self.id = int(id)
        self.name = name
        self.supercategory = supercategory if supercategory else name

    @classmethod
    def from_coco_category(cls, category):
        """
        Creates CocoCategory object using coco category.

        Args:
            category: Dict
                {"supercategory": "person", "id": 1, "name": "person"},
        """
        return cls(
            id=category["id"],
            name=category["name"],
            supercategory=category["supercategory"] if "supercategory" in category else category["name"],
        )

    @property
    def json(self):
        return {
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
        }

    def __repr__(self):
        return f"""CocoCategory<
    id: {self.id},
    name: {self.name},
    supercategory: {self.supercategory}>"""


class CocoAnnotation:
    """
    COCO formatted annotation.
    """

    @classmethod
    def from_coco_segmentation(cls, segmentation, category_id, category_name, iscrowd=0):
        """
        Creates CocoAnnotation object using coco segmentation.

        Args:
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            iscrowd: int
                0 or 1
        """
        return cls(
            segmentation=segmentation,
            category_id=category_id,
            category_name=category_name,
            iscrowd=iscrowd,
        )

    @classmethod
    def from_coco_bbox(cls, bbox, category_id, category_name, iscrowd=0):
        """
        Creates CocoAnnotation object using coco bbox

        Args:
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            iscrowd: int
                0 or 1
        """
        return cls(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            iscrowd=iscrowd,
        )

    @classmethod
    def from_coco_annotation_dict(cls, annotation_dict: Dict, category_name: Optional[str] = None):
        """
        Creates CocoAnnotation object from category name and COCO formatted
        annotation dict (with fields "bbox", "segmentation", "category_id").

        Args:
            category_name: str
                Category name of the annotation
            annotation_dict: dict
                COCO formatted annotation dict (with fields "bbox", "segmentation", "category_id")
        """
        if annotation_dict.__contains__("segmentation") and not isinstance(annotation_dict["segmentation"], list):
            has_rle_segmentation = True
            logger.warning(
                f"Segmentation annotation for id {annotation_dict['id']} is skipped since RLE segmentation format is not supported."
            )
        else:
            has_rle_segmentation = False

        if (
            annotation_dict.__contains__("segmentation")
            and annotation_dict["segmentation"]
            and not has_rle_segmentation
        ):
            return cls(
                segmentation=annotation_dict["segmentation"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
            )
        else:
            return cls(
                bbox=annotation_dict["bbox"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
            )

    @classmethod
    def from_shapely_annotation(
        cls,
        shapely_annotation: ShapelyAnnotation,
        category_id: int,
        category_name: str,
        iscrowd: int,
    ):
        """
        Creates CocoAnnotation object from ShapelyAnnotation object.

        Args:
            shapely_annotation (ShapelyAnnotation)
            category_id (int): Category id of the annotation
            category_name (str): Category name of the annotation
            iscrowd (int): 0 or 1
        """
        coco_annotation = cls(
            bbox=[0, 0, 0, 0],
            category_id=category_id,
            category_name=category_name,
            iscrowd=iscrowd,
        )
        coco_annotation._segmentation = shapely_annotation.to_coco_segmentation()
        coco_annotation._shapely_annotation = shapely_annotation
        return coco_annotation

    def __init__(
        self,
        segmentation=None,
        keypoints=None,
        bbox=None,
        category_id=None,
        category_name=None,
        image_id=None,
        iscrowd=0,
    ):
        """
        Creates coco annotation object using bbox or segmentation
        keypoints format: [x y v x y v x y v] where v is visible (0 none, 1 labeled but not visibile, 2 labeled and visible)
        Args:
            keypoints: List[List]
                [[121, 263, 2, 121, 350, 2, 448, 312, 2]]
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            image_id: int
                Image ID of the annotation
            iscrowd: int
                0 or 1
        """
        if bbox is None and segmentation is None:
            raise ValueError("you must provide a bbox or polygon")

        self._keypoints = keypoints
        self._segmentation = segmentation
        bbox = [round(point) for point in bbox] if bbox else bbox
        self._category_id = category_id
        self._category_name = category_name
        self._image_id = image_id
        self._iscrowd = iscrowd

        if self._segmentation:
            shapely_annotation = ShapelyAnnotation.from_coco_segmentation(segmentation=self._segmentation)
        else:
            shapely_annotation = ShapelyAnnotation.from_coco_bbox(bbox=bbox)
        self._shapely_annotation = shapely_annotation

    def get_sliced_coco_annotation(self, slice_bbox: List[int]):
        shapely_polygon = box(slice_bbox[0], slice_bbox[1], slice_bbox[2], slice_bbox[3])
        intersection_shapely_annotation = self._shapely_annotation.get_intersection(shapely_polygon)
        return CocoAnnotation.from_shapely_annotation(
            intersection_shapely_annotation,
            category_id=self.category_id,
            category_name=self.category_name,
            iscrowd=self.iscrowd,
        )

    @property
    def area(self):
        """
        Returns area of annotation polygon (or bbox if no polygon available)
        """
        return self._shapely_annotation.area

    @property
    def bbox(self):
        """
        Returns coco formatted bbox of the annotation as [xmin, ymin, width, height]
        """
        return self._shapely_annotation.to_coco_bbox()

    @property
    def keypoints(self):
        """
        Return coco formatted keypoints of the annotation as [[121, 263, 2, 121, 350, 2, 448, 312, 2]]
        """
        return self._keypoints

    @property
    def segmentation(self):
        """
        Returns coco formatted segmentation of the annotation as [[1, 1, 325, 125, 250, 200, 5, 200]]
        """
        if self._segmentation:
            return self._shapely_annotation.to_coco_segmentation()
        else:
            return []

    @property
    def category_id(self):
        """
        Returns category id of the annotation as int
        """
        return self._category_id

    @category_id.setter
    def category_id(self, i):
        if not isinstance(i, int):
            raise Exception("category_id must be an integer")
        self._category_id = i

    @property
    def image_id(self):
        """
        Returns image id of the annotation as int
        """
        return self._image_id

    @image_id.setter
    def image_id(self, i):
        if not isinstance(i, int):
            raise Exception("image_id must be an integer")
        self._image_id = i

    @property
    def category_name(self):
        """
        Returns category name of the annotation as str
        """
        return self._category_name

    @category_name.setter
    def category_name(self, n):
        if not isinstance(n, str):
            raise Exception("category_name must be a string")
        self._category_name = n

    @property
    def iscrowd(self):
        """
        Returns iscrowd info of the annotation
        """
        return self._iscrowd

    @property
    def json(self):
        return {
            "image_id": self.image_id,
            "bbox": self.bbox,
            "category_id": self.category_id,
            "keypoints": self.keypoints,
            "segmentation": self.segmentation,
            "iscrowd": self.iscrowd,
            "area": self.area,
        }

    def serialize(self):
        print(".serialize() is deprectaed, use .json instead")

    def __repr__(self):
        return f"""CocoAnnotation<
    image_id: {self.image_id},
    bbox: {self.bbox},
    keypoints: {self.keypoints},
    segmentation: {self.segmentation},
    category_id: {self.category_id},
    category_name: {self.category_name},
    iscrowd: {self.iscrowd},
    area: {self.area}>"""



def create_coco_dict(images, categories, ignore_negative_samples=False, image_id_setting="auto"):
    """
    Creates COCO dict with fields "images", "annotations", "categories".

    Arguments
    ---------
        images : List of CocoImage containing a list of CocoAnnotation
        categories : List of Dict
            COCO categories
        ignore_negative_samples : Bool
            If True, images without annotations are ignored
        image_id_setting: str
            how to assign image ids while exporting can be
                auto --> will assign id from scratch (<CocoImage>.id will be ignored)
                manual --> you will need to provide image ids in <CocoImage> instances (<CocoImage>.id can not be None)
    Returns
    -------
        coco_dict : Dict
            COCO dict with fields "images", "annotations", "categories"
    """
    # assertion of parameters
    if image_id_setting not in ["auto", "manual"]:
        raise ValueError(f"'image_id_setting' should be one of ['auto', 'manual']")

    # define accumulators
    image_index = 1
    annotation_id = 1
    coco_dict = dict(images=[], annotations=[], categories=categories)
    for coco_image in images:
        # get coco annotations
        coco_annotations = coco_image.annotations
        # get num annotations
        num_annotations = len(coco_annotations)
        # if ignore_negative_samples is True and no annotations, skip image
        if ignore_negative_samples and num_annotations == 0:
            continue
        else:
            # get image_id
            if image_id_setting == "auto":
                image_id = image_index
                image_index += 1
            elif image_id_setting == "manual":
                if coco_image.id is None:
                    raise ValueError("'coco_image.id' should be set manually when image_id_setting == 'manual'")
                image_id = coco_image.id

            # create coco image object
            out_image = {
                "height": coco_image.height,
                "width": coco_image.width,
                "id": image_id,
                "file_name": coco_image.file_name,
            }
            coco_dict["images"].append(out_image)

            # do the same for image annotations
            for coco_annotation in coco_annotations:
                # create coco annotation object
                out_annotation = {
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": coco_annotation.bbox,
                    "keypoints": coco_annotation.keypoints,
                    "segmentation": coco_annotation.segmentation,
                    "category_id": coco_annotation.category_id,
                    "id": annotation_id,
                    "area": coco_annotation.area,
                }
                coco_dict["annotations"].append(out_annotation)
                # increment annotation id
                annotation_id += 1

    # return coco dict
    return coco_dict
