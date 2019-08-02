import os
import sys
import random
import skimage.io
from mrcnn.config import Config
from datetime import datetime

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_0030.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


# import train_tongue
# class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'car', 'leg', 'well']
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

a = datetime.now()
# Run detection
results = model.detect([image], verbose=1)
b = datetime.now()
# Visualize results
print("shijian", (b - a).seconds)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])



#计算单张f1_measure
def compute_f1_measure(local,ground_truth):
    overlap_area=0
    mask_area=0
    FP=0
    FN=0
    for i in range(800):
        for j in range(1280):
            if ground_truth[i][j]:
                mask_area+=1
            for k in range (local.shape[2]):
                if local[i][j][k] == ground_truth[i][j] and ground_truth[i][j] :
                    overlap_area+=1
                if local[i][j][k] and ground_truth[i][j] != local[i][j][k]:
                    FP+=1
                if local[i][j][k] != ground_truth[i][j] and ground_truth[i][j]:
                    FN+=1
    print ("overlap_area",overlap_area)
    print ("mask_area:",mask_area)
    TP=overlap_area
    P=TP/(TP+FP)
    R=TP/(TP+FN)
    f1_measure=2*P*R/(P+R)
    return f1_measure
#计算单张mAP值
def compute_mAP(local,ground_truth):
    overlap_area=0
    mask_area=0
    FP=0
    FN=0
    for i in range(800):
        for j in range(1280):
            if ground_truth[i][j]:
                mask_area+=1
            for k in range (local.shape[2]):
                if local[i][j][k] == ground_truth[i][j] and ground_truth[i][j] :
                    overlap_area+=1
                if local[i][j][k] and ground_truth[i][j] != local[i][j][k]:
                    FP+=1
                if local[i][j][k] != ground_truth[i][j] and ground_truth[i][j]:
                    FN+=1
    print ("overlap_area",overlap_area)
    print ("mask_area:",mask_area)
    TP=overlap_area
    P=TP/(TP+FP)
    #R=TP/(TP+FN)
    #f1_measure=2*P*R/(P+R)
    return P