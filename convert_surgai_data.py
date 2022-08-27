import labelme2coco
# set directory that contains labelme annotations and image files
labelme_folder = "/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/raw_images_resized_1500/1/Frames"
# labelme_folder = "../data/Hadassah_2021-07-05/raw_images_debug_resized/3/Frames"

# set export dir
export_dir = "/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/for_centernet/annotations_1500"

# set train split rate
train_split_rate = 1.0

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, export_dir, train_split_rate)