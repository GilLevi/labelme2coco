import json
from PIL import Image

annotation_file = "../data/Hadassah_2021-07-05/raw_images_debug/3/Frames/frame0067.json"
image_path = "../data/Hadassah_2021-07-05/raw_images_debug/3/Frames/frame0067.png"

resize_annotation_file = "../data/Hadassah_2021-07-05/raw_images_debug_1500/3/Frames/frame0067.json"
resize_image_path = "../data/Hadassah_2021-07-05/raw_images_debug_1500/3/Frames/frame0067.png"

OUT_SIZE = 1500

with open(annotation_file) as f:
    ann = json.load(f)

im = Image.open(image_path)
width, height = im.size
# ann is organized as (x,y)

for ann_shape in ann['shapes']:
    for i in range(len(ann_shape['points'])):
        ann_shape['points'][i][0] *= (OUT_SIZE / width)
        ann_shape['points'][i][1] *= (OUT_SIZE / height)


with open(resize_annotation_file, 'w+') as f:
    json.dump(ann, f)
im = im.resize((OUT_SIZE, OUT_SIZE))
im.save(resize_image_path)
gil = 1
