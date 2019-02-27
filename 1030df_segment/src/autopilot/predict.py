import argparse
import datetime
import os
import sys
import time

import numpy as np
import skimage.io
import tensorflow as tf

os.chdir(os.path.dirname(__file__))

ROOT_DIR = os.path.abspath('../../')
sys.path.append(os.path.join(ROOT_DIR, 'src'))

# from autopilot.train_autopilot import CocoConfig
from mrcnn import model as modellib
from mrcnn.config import Config

coco_class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

coco_class_names[coco_class_names.index('motorcycle')] = 'motorbicycle'

apollo_class_names = ['BG', 'car', 'motorbicycle', 'bicycle', 'person', 'truck', 'bus', 'tricycle']

label_to_name = {33: 'car',
                 34: 'motorbicycle', 35: 'bicycle', 36: 'person',
                 38: 'truck', 39: 'bus', 40: 'tricycle'}

apollo_class_name_to_label = {v: k for k, v in label_to_name.items()}


class InferenceConfig(Config):
    NAME = "coco"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0

    # TODO
    NUM_CLASSES = 1 + 80  # COCO has 80 classes
    # IMAGE_RESIZE_MODE = "none"


def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]

    return ''.join(map(lambda x, y: ''.join([str(x), ' ', str(y), '|']), rle[:, 0], rle[:, 1]))


def mask_to_rle(image_id, rois, class_ids, scores, masks):
    if class_ids.shape[-1] == 0:
        # TODO
        # prediction_file.write(image_id + ',' + '33, 1, 100,1 100|\n')
        return "{},{},{},{},{}".format(image_id, '33', '1', '100', '1 100|')

    n_instances = masks.shape[-1]

    print(n_instances)

    lines = []
    for n in range(n_instances):
        class_id = class_ids[n]
        class_name = coco_class_names[class_id]
        print(n, class_id, class_name)
        if class_name not in apollo_class_names:
            continue
        if class_name == 'BG':
            continue
        label_id = apollo_class_name_to_label[class_name]
        confidence = scores[n]
        mask = masks[:, :, n]
        pixel_count = mask.sum()
        encoded_pixels = rle_encode(mask)
        lines.append("{},{},{},{},{}".format(image_id, label_id, pixel_count, confidence, encoded_pixels))

    if len(lines) == 0:
        return "{},{},{},{},{}".format(image_id, '33', '1', '100', '1 100|')

    return "\n".join(lines)


def detect(model, image_dir=None, limit=None):
    submission = []
    for root, dirs, image_files in sorted(os.walk(os.path.expanduser(image_dir))):
        if limit:
            limit = int(limit)
            image_files = image_files[:limit]
        for index, image_file in enumerate(image_files):
            print('=' * 80)
            print(index, image_file)

            image = skimage.io.imread(os.path.join(root, image_file))
            # Detect objects
            r = model.detect([image], verbose=0)[0]

            image_id = image_file[:-4]
            rois, class_ids, scores, masks = r["rois"], r["class_ids"], r["scores"], r["masks"].astype(np.uint8)
            rle = mask_to_rle(image_id, rois, class_ids, scores, masks)

            submission.append(rle)

    submission = "ImageId,LabelId,PixelCount,Confidence,EncodedPixels\n" + "\n".join(submission)

    file_path = "../../submissions/submit_{:%Y%m%d_%H%M%S}.csv".format(datetime.datetime.now())
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", file_path)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser(
        description='Detect By Mask R-CNN')

    # noinspection PyPackageRequirements
    parser.add_argument('--CUDA_VISIBLE_DEVICES', required=False,
                        default=None,
                        metavar="gpuid1,gpuid2",
                        help='cuda visible devices')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image_dir', required=True,
                        metavar="/path/to/image dir/",
                        help='image directory')
    parser.add_argument('--limit', required=False,
                        default=None,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    if args.CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.CUDA_VISIBLE_DEVICES).split(',')[0]

    print("CUDA_VISIBLE_DEVICES: ", args.CUDA_VISIBLE_DEVICES)
    # GPU config
    config_env = tf.ConfigProto()
    #  allocate gpu memory as needed, sess = tf.Session(config=config) KTF.set_session(sess)
    config_env.gpu_options.allow_growth = True

    # Configurations
    config = InferenceConfig()
    config.display()

    model_path = os.path.expanduser(args.model)
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=model_path)

    # TODO
    model.load_weights(model_path, by_name=True)

    detect(model, image_dir=args.image_dir, limit=args.limit)

    print('Elapsed time', round((time.time() - start) / 60, 1), 'minutes')

