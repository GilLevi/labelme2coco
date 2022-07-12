import json

#with open("../data/Hadassah_2021-07-05/labelme2coco_data/dataset.json") as f:
with open("../data/Hadassah_2021-07-05/labelme2coco_data/coco_format_tests/dataset.json") as f:
        coco_ann = json.load(f)


# compare to the labelme annotations for the same image:
with open("/Users/gillevi/Projects/SurgeonAI/data/Hadassah_2021-07-05/raw_images/1/Frames/frame3520.json") as f:
    labelme_ann = json.load(f)
print('label me format keypoint annotations')
print(labelme_ann['shapes'][0]['points'])
print(labelme_ann['shapes'][1]['points'])
print()
# now, let's check the coco skeleton annotation format
with open("../data/Hadassah_2021-07-05/labelme2coco_data/coco_dataset_annotations/person_keypoints_val2017.json") as f:
    org_coco_ann = json.load(f)

print('original coco annotations keypoints example:')
print(org_coco_ann['annotations'][0]['keypoints'])
print()
# let's consider image frame3520.png , the image id in our data is 1 and it has 2 annotations associated with it
#print(coco_ann['images'][0])
print('my converted annotations keypoints')
print(coco_ann['annotations'][0]['keypoints'])
print()
# COCO keypoints format
# keypoints" is a length 3*17 array (x, y, v) for body keypoints. Each keypoint has a 0-indexed location x,y and a
# visibility flag v defined as v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible.
# A keypoint is considered visible if it falls inside the object segment.
#
# "num_keypoints" indicates the number of labeled body keypoints (v>0), (e.g. crowds and small objects, will have num_keypoints=0).

gil = 1
