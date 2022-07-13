import labelme2coco
# set directory that contains labelme annotations and image files
labelme_folder = "../data/Hadassah_2021-07-05/raw_images/4/Frames"

# set export dir
export_dir = "../data/Hadassah_2021-07-05/labelme2coco_data/surgai_annotations_coco_format"

# set train split rate
train_split_rate = 1.0

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, export_dir, train_split_rate)