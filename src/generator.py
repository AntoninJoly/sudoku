from keras.utils import np_utils
import numpy as np
import tifffile
from glob import glob
import os
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, RandomSizedCrop,\
                           RandomCrop, ChannelShuffle, GlassBlur, GridDistortion, PadIfNeeded, \
                           CropNonEmptyMaskIfExists
import cv2
import sys
import json
import warnings
warnings.filterwarnings("ignore")

def batch_generator(img_dir, batch_size, seed, flag):
    paths = get_data_paths(img_dir, seed)
    steps = len(paths) // batch_size
    batch = generate_batch_data(paths, batch_size, flag)
    return batch, steps

def get_data_paths(img_dir, seed=37):
    img_paths = os.listdir(img_dir)
    np.random.seed(seed)
    np.random.shuffle(img_paths)
    return img_paths

def generate_batch_data(img_paths, batch_size, flag):
    while 1:
        for i in range(0, len(img_paths), batch_size):
            idx_start = 0 if (i + batch_size) > len(img_paths) else i
            idx_end = idx_start + batch_size
            if flag == 'train':
                images, gts = read_train_img(img_paths[idx_start: idx_end])
            elif flag == 'val':
                images, gts = read_val_img(img_paths[idx_start: idx_end])
            yield (images, gts)

def read_train_img(images_paths):
    images = []
    gts = []
    for image_path in images_paths:
        image = cv2.imread(os.path.join('data/train/img',image_path))
        gt_path = image_path.replace('jpg','json')
        gt_polygon = json.loads(open(os.path.join(os.getcwd(),'data/train/label',gt_path), "r").read())

        gt = np.zeros(image.shape)
        for i in gt_polygon['shapes']:
            pts = np.array(i['points'], np.int32).reshape((-1,1,2))
            gt = cv2.fillPoly(gt,[pts], 255)
        
        aug = Compose([VerticalFlip(),
                       RandomRotate90(),
                       HorizontalFlip(),
                       ChannelShuffle(),
                       GlassBlur(),
                       GridDistortion(),
                       PadIfNeeded(min_height=480, min_width=640, p=1),
                       CropNonEmptyMaskIfExists(height=480, width=640, p=1)])

        augmented = aug(image=image, mask=gt)
        
        image = augmented['image'] / 255.0
        gt = augmented['mask'] / 255.0
        images.append(image)
        gts.append(gt)
    
    return np.array(images), np.array(gts)

def read_val_img(images_paths):
    images = []
    gts = []
    for image_path in images_paths:
        image = cv2.imread(os.path.join('data/val/img',image_path))
        gt_path = image_path.replace('jpg','json')
        gt_polygon = json.loads(open(os.path.join(os.getcwd(),'data/val/label',gt_path), "r").read())

        gt = np.zeros(image.shape)
        for i in gt_polygon['shapes']:
            pts = np.array(i['points'], np.int32).reshape((-1,1,2))
            gt = cv2.fillPoly(gt,[pts], 255)
        
        aug = Compose([PadIfNeeded(min_height=480, min_width=640, p=1),
                       CropNonEmptyMaskIfExists(height=480, width=640, p=1)])
        augmented = aug(image=image, mask=gt)
        
        image = augmented['image'] / 255.0
        gt = augmented['mask'] / 255.0

        images.append(image)
        gts.append(gt)

    return np.array(images), np.array(gts)