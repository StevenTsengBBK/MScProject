import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from shutil import copyfile

import torch
import torch.nn as nn

import STResNeSt.encoding as encoding

DOWNLOAD_DIR = os.path.expanduser("./MFCC")
DATASET_DIR = os.path.expanduser("~/encoding/data")
class_label = ["air_conditioner"
, "car_horn"
, "children_playing"
, "dog_bark"
, "drilling"
, "engine_idling"
, "gun_shot"
, "jackhammer"
, "siren"
, "street_music"]


# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []

# Setting the random seed of PyTorch
torch.manual_seed(10)
torch.cuda.manual_seed(10)

# Dataloader
# Load dataset in class subfolders
if os.path.exists(DATASET_DIR):
    print("Datasets prepared. Skip preparation step.")
else:
    train_files = os.listdir(DOWNLOAD_DIR + '/urbansound8k_img_train')
    test_files = os.listdir(DOWNLOAD_DIR + '/urbansound8k_img_test')
    os.makedirs(DATASET_DIR + "/urbansound8k/train")
    os.makedirs(DATASET_DIR + "/urbansound8k/val")
    for l in class_label:
        os.makedirs(DATASET_DIR + "/urbansound8k/train/" + l)
        os.makedirs(DATASET_DIR + "/urbansound8k/val/" + l)
    print("Train set importing...")
    for file in train_files:
        if not file.startswith('.'):
            class_id = file.split("-")[1]
            label = class_label[int(class_id)]
            copyfile(DOWNLOAD_DIR + '/urbansound8k_img_train/'+file, DATASET_DIR + "/urbansound8k/train/" + label + "/" + file)
    print("Train set imported")
    print("Test set importing...")
    for file in test_files:
        if not file.startswith('.'):
            class_id = file.split("-")[1]
            label = class_label[int(class_id)]
            copyfile(DOWNLOAD_DIR + '/urbansound8k_img_test/'+ file, DATASET_DIR + "/urbansound8k/val/" + label + "/"+ file)
    print("Test set imported")


transform_train, transform_val = encoding.transforms.get_transform(
            'imagenet', None, 224, False)
trainset = encoding.datasets.get_dataset('imagenet', root=os.path.expanduser('~/encoding/data'),
                                             transform=transform_train, train=True, download=True)
valset = encoding.datasets.get_dataset('imagenet', root=os.path.expanduser('~/encoding/data'),
                                           transform=transform_val, train=False, download=True)
