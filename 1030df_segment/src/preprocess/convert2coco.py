import datetime
import fnmatch
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

import coco_util

INFO = {
    "description": "Example Dataset",
    "url": "",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]


def gen_categories():
    label = pd.read_csv(LABEL_CSV_PATH, header=None, names=['name', 'id'])
    label['supercategory'] = 'autopilot'

    cat_ids = [33, 34, 35, 36, 38, 39, 40]
    label = label[label['id'].apply(lambda cat_id: cat_id in cat_ids)]
    return label.to_dict(orient='records')


def gen_annotations_json_file(categories, image_dir, annotation_dir, image_size=(1024, 1024)):
    coco_output = {
        "info": INFO,
        "licenses": None,
        "categories": categories,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    cat_ids = [cat['id'] for cat in categories]

    # filter for jpeg images
    for root, _, files in os.walk(image_dir):
        image_files = filter_for_jpeg(root, files)
        image_files.sort()

        # go through each image
        for index, image_filename in enumerate(image_files):
            print(index, image_filename)

            annotation_basename = os.path.basename(image_filename)[0:-4] + '_instanceIds.png'
            annotation_filename = os.path.join(annotation_dir, annotation_basename)

            ann_png = Image.open(annotation_filename)
            binary_ann_png = np.asarray(ann_png)

            unique_pixes = np.unique(binary_ann_png)

            target_pixes = [int(pix) for pix in unique_pixes if np.floor_divide(pix, 1000) in cat_ids]
            if len(target_pixes) == 0:
                print('skipped')
            else:
                image_info = coco_util.create_image_info(image_id,
                                                         os.path.basename(image_filename),
                                                         image_size,
                                                         date_captured='')

                coco_output["images"].append(image_info)

                for i, target_pix in enumerate(target_pixes):
                    class_id = int(target_pix / 1000)
                    category_info = {'id': class_id, 'is_crowd': 0}

                    binary_mask = np.where(binary_ann_png == target_pix, 1, 0).astype(np.uint8)
                    annotation_info = coco_util.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image_size, tolerance=2)

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)
                        segmentation_id = segmentation_id + 1

                image_id = image_id + 1

    return coco_output


LABEL_CSV_PATH = os.path.join(ROOT_DIR, 'data/label.csv')

# DATA_DIR = ROOT_DIR
# IMAGE_DIR = os.path.join(DATA_DIR, "data/train_color")
# ANNOTATION_DIR = os.path.join(DATA_DIR, "data/train_label")


DATA_DIR = os.path.expanduser('~/Documents/ml_data/datafountain')
IMAGE_DIR = os.path.join(DATA_DIR, "image_output")
ANNOTATION_DIR = os.path.join(DATA_DIR, "label_output")


def main():
    categories = gen_categories()
    # image_size = (3384, 2710)
    image_size = (1024, 1024)
    coco_obj = gen_annotations_json_file(categories, IMAGE_DIR, ANNOTATION_DIR, image_size=image_size)
    annotation_file = 'instances_all'

    with open('{}/annotations/{}.json'.format(ROOT_DIR, annotation_file), 'w') as output_json_file:
        json.dump(coco_obj, output_json_file)


if __name__ == "__main__":
    main()
