##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yao-I Tseng
## Email: mrsuccess1203@gmail.com
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from shutil import copyfile
import numpy as np

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

IMAGE_TYPE = ["MFCC", "STFT"]
DOWNLOAD_DIR = os.path.expanduser("./Colour_MFCC")
DATASET_DIR = os.path.expanduser("./encoding/data")

CLASS1_LABELID = 4
CLASS2_LABELID = 5

def DataPrepare():
    # Load dataset in class subfolders
    if os.path.exists(DATASET_DIR):
        print("Datasets prepared. Skip preparation step.")
    else:
        print("Standard Dataset Importing.")

        # Create Directories
        os.makedirs(DATASET_DIR + "/urbansound8k/train")
        os.makedirs(DATASET_DIR + "/urbansound8k/val")
        os.makedirs(DATASET_DIR + "/urbansound8k/test/")
        # for l in class_label:
        #     os.makedirs(DATASET_DIR + "/urbansound8k/train/" + l)
        #     os.makedirs(DATASET_DIR + "/urbansound8k/val/" + l)
        #     os.makedirs(DATASET_DIR + "/urbansound8k/test/" + l)
        os.makedirs(DATASET_DIR + "/urbansound8k/train/" + class_label[CLASS1_LABELID])
        os.makedirs(DATASET_DIR + "/urbansound8k/val/" + class_label[CLASS1_LABELID])
        os.makedirs(DATASET_DIR + "/urbansound8k/test/" + class_label[CLASS1_LABELID])
        os.makedirs(DATASET_DIR + "/urbansound8k/train/" + class_label[CLASS2_LABELID])
        os.makedirs(DATASET_DIR + "/urbansound8k/val/" + class_label[CLASS2_LABELID])
        os.makedirs(DATASET_DIR + "/urbansound8k/test/" + class_label[CLASS2_LABELID])

        # Classifying and split into train and test set

        for fold in range(1, 11):
            fileList = os.listdir(DOWNLOAD_DIR + '/fold' + str(fold))
            if fold in [8, 9]:
                print("Validation set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     DATASET_DIR + "/urbansound8k/val/" + label + "/" + file)
                print("Validation set imported")
            elif fold == 10:
                print("Test set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     DATASET_DIR + "/urbansound8k/test/" + label + "/" + file)
                print("Test set imported")
            else:
                print("Train set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     DATASET_DIR + "/urbansound8k/train/" + label + "/" + file)
                print("Train set imported")
    print("Standard Dataset Import Succeeded.")

# Mini Dataset version is used for reducing training time
# Each class will have at most 100 sample in training set and 10 samples in test set
def MiniDataPrepare():
    train_set_class_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    test_set_class_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Load dataset in class subfolders
    if os.path.exists(DATASET_DIR):
        print("Datasets prepared. Skip preparation step.")
    else:
        print("Mini Dataset Importing.")

        # Create Directories
        os.makedirs(DATASET_DIR + "/urbansound8k/train")
        os.makedirs(DATASET_DIR + "/urbansound8k/val")
        os.makedirs(DATASET_DIR + "/urbansound8k/test/")
        # for l in class_label:
        #     os.makedirs(DATASET_DIR + "/urbansound8k/train/" + l)
        #     os.makedirs(DATASET_DIR + "/urbansound8k/val/" + l)
        #     os.makedirs(DATASET_DIR + "/urbansound8k/test/" + l)
        os.makedirs(DATASET_DIR + "/urbansound8k/train/" + class_label[CLASS1_LABELID])
        os.makedirs(DATASET_DIR + "/urbansound8k/val/" + class_label[CLASS1_LABELID])
        os.makedirs(DATASET_DIR + "/urbansound8k/test/" + class_label[CLASS1_LABELID])
        os.makedirs(DATASET_DIR + "/urbansound8k/train/" + class_label[CLASS2_LABELID])
        os.makedirs(DATASET_DIR + "/urbansound8k/val/" + class_label[CLASS2_LABELID])
        os.makedirs(DATASET_DIR + "/urbansound8k/test/" + class_label[CLASS2_LABELID])

        # Classifying and split into train and test set
        for fold in range(1, 11):
            fileList = os.listdir(DOWNLOAD_DIR + '/fold' + str(fold))
            if fold in [8, 9]:
                print("Validation set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            if not test_set_class_count[class_id] == 10:
                                label = class_label[class_id]
                                copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                         DATASET_DIR + "/urbansound8k/val/" + label + "/" + file)
                                test_set_class_count[int(class_id)] += 1
                print("Validation set imported")
            elif fold == 10:
                print("Test set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     DATASET_DIR + "/urbansound8k/test/" + label + "/" + file)
                print("Test set imported")
            else:
                print("Train set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            if not train_set_class_count[class_id] == 100:
                                label = class_label[class_id]
                                copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                         DATASET_DIR + "/urbansound8k/train/" + label + "/" + file)
                                train_set_class_count[int(class_id)] += 1
                print("Train set imported")

    print("Mini Dataset Import Succeeded.")

def DataPrepareFiveFold():
    # Create Directories
    if os.path.exists(DATASET_DIR):
        print("Datasets prepared. Skip preparation step.")
        return
    else:
        print("Five Fold Dataset Importing.")

    # Load dataset in class subfolders
    CV_sets = np.array([1])
    validation_sets = np.array([6,7])

    for iteration in range(1,6):
        ITER_DIR = DATASET_DIR + "/round" + str(iteration) + "/urbansound8k/"
        os.makedirs(ITER_DIR +"/train")
        os.makedirs(ITER_DIR +"/val")
        os.makedirs(ITER_DIR +"/test")
        os.makedirs(ITER_DIR + "/CV")
        os.makedirs(ITER_DIR +"/train/" + class_label[CLASS1_LABELID])
        os.makedirs(ITER_DIR +"/val/" + class_label[CLASS1_LABELID])
        os.makedirs(ITER_DIR +"/test/" + class_label[CLASS1_LABELID])
        os.makedirs(ITER_DIR + "/CV/" + class_label[CLASS1_LABELID])
        os.makedirs(ITER_DIR +"/train/" + class_label[CLASS2_LABELID])
        os.makedirs(ITER_DIR +"/val/" + class_label[CLASS2_LABELID])
        os.makedirs(ITER_DIR +"/test/" + class_label[CLASS2_LABELID])
        os.makedirs(ITER_DIR + "/CV/" + class_label[CLASS2_LABELID])

        TRAIN_DIR = ITER_DIR +"/train"
        VAL_DIR = ITER_DIR +"/val"
        TEST_DIR = ITER_DIR + "/test"
        CV_DIR = ITER_DIR + "/CV"

        # Classifying and split into train and test set
        for fold in range(1, 11):
            fileList = os.listdir(DOWNLOAD_DIR + '/fold' + str(fold))
            if fold in validation_sets:
                print("Validation set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     VAL_DIR + "/" + label + "/" + file)
                print("Validation set imported")
            elif fold in CV_sets:
                print("CV set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     CV_DIR + "/" + label + "/" + file)
                print("CV set imported")
            elif fold < 6:
                print("Train set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     TRAIN_DIR + "/" + label + "/" + file)
                print("Train set imported")
            else:
                print("Test set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     TEST_DIR + "/" + label + "/" + file)
                print("Test set imported")
        CV_sets = CV_sets + 1
    print("Five Fold Dataset Import Succeeded.")