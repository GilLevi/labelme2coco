import labelme2coco
# set directory that contains labelme annotations and image files
labelme_folder = "../data/Hadassah_2021-07-05/raw_images/1/Frames"

# set export dir
export_dir = "../data/Hadassah_2021-07-05/labelme2coco_data/coco_format_tests"

# set train split rate
train_split_rate = 0.85

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, export_dir, train_split_rate)