##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yao-I Tseng
## Email: mrsuccess1203@gmail.com
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from shutil import copyfile
import numpy as np
import shutil
import time

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

DATASET_DIR = os.path.expanduser("./encoding/data")

def DataPrepare(CLASS1_LABELID, CLASS2_LABELID, Download_folder):
    DOWNLOAD_DIR = os.path.expanduser("./" + Download_folder)
    
    # Load dataset in class subfolders
    if os.path.exists(DATASET_DIR):
        print("Directory Existed. Deleting Directory!")
        shutil.rmtree(DATASET_DIR)

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
            for file in fileList:
                if not file.startswith('.'):
                    class_id = int(file.split("-")[1])
                    if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                        label = class_label[class_id]
                        copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                 DATASET_DIR + "/urbansound8k/val/" + label + "/" + file)
        elif fold == 10:
            for file in fileList:
                if not file.startswith('.'):
                    class_id = int(file.split("-")[1])
                    if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                        label = class_label[class_id]
                        copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                 DATASET_DIR + "/urbansound8k/test/" + label + "/" + file)
        else:
            for file in fileList:
                if not file.startswith('.'):
                    class_id = int(file.split("-")[1])
                    if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                        label = class_label[class_id]
                        copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                 DATASET_DIR + "/urbansound8k/train/" + label + "/" + file)
    
    timestamp()
    print("Standard Dataset Import Succeeded.")

# Mini Dataset version is used for reducing training time
# Each class will have at most 100 sample in training set and 10 samples in test set
def MiniDataPrepare(CLASS1_LABELID, CLASS2_LABELID, Download_folder):
    DOWNLOAD_DIR = os.path.expanduser("./" + Download_folder)
    
    train_set_class_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    test_set_class_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Load dataset in class subfolders
    if os.path.exists(DATASET_DIR):
        print("Directory Existed. Deleting Directory!")
        shutil.rmtree(DATASET_DIR)
    
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
            for file in fileList:
                if not file.startswith('.'):
                    class_id = int(file.split("-")[1])
                    if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                        if not test_set_class_count[class_id] == 10:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     DATASET_DIR + "/urbansound8k/val/" + label + "/" + file)
                            test_set_class_count[int(class_id)] += 1
        elif fold == 10:
            for file in fileList:
                if not file.startswith('.'):
                    class_id = int(file.split("-")[1])
                    if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                        label = class_label[class_id]
                        copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                 DATASET_DIR + "/urbansound8k/test/" + label + "/" + file)
        else:
            for file in fileList:
                if not file.startswith('.'):
                    class_id = int(file.split("-")[1])
                    if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                        if not train_set_class_count[class_id] == 100:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     DATASET_DIR + "/urbansound8k/train/" + label + "/" + file)
                            train_set_class_count[int(class_id)] += 1
    
    timestamp()
    print("Mini Dataset Import Succeeded.")

def DataPrepareFiveFold(CLASS1_LABELID, CLASS2_LABELID, Download_folder):
    DOWNLOAD_DIR = os.path.expanduser("./" + Download_folder)
    
    # Create Directories
    if os.path.exists(DATASET_DIR):
        print("Directory Existed. Deleting Directory!")
        shutil.rmtree(DATASET_DIR)
    
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
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     VAL_DIR + "/" + label + "/" + file)
            elif fold in CV_sets:
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     CV_DIR + "/" + label + "/" + file)
            elif fold < 6:
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     TRAIN_DIR + "/" + label + "/" + file)
            else:
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                            label = class_label[class_id]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     TEST_DIR + "/" + label + "/" + file)
        CV_sets = CV_sets + 1
    
    timestamp()
    print("Five Fold Dataset Import Succeeded.")
    
def FullDataPrepareFiveFold(Download_folder):
    DOWNLOAD_DIR = os.path.expanduser("./" + Download_folder)
    
    # Create Directories
    if os.path.exists(DATASET_DIR):
        print("Directory Existed. Deleting Directory!")
        shutil.rmtree(DATASET_DIR)
    
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
        for label in class_label:
            os.makedirs(ITER_DIR +"/train/" + label)
            os.makedirs(ITER_DIR +"/val/" + label)
            os.makedirs(ITER_DIR +"/test/" + label)
            os.makedirs(ITER_DIR + "/CV/" + label)

        TRAIN_DIR = ITER_DIR +"/train"
        VAL_DIR = ITER_DIR +"/val"
        TEST_DIR = ITER_DIR + "/test"
        CV_DIR = ITER_DIR + "/CV"

        # Classifying and split into train and test set
        for fold in range(1, 11):
            fileList = os.listdir(DOWNLOAD_DIR + '/fold' + str(fold))
            if fold in validation_sets:
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        label = class_label[class_id]
                        copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                 VAL_DIR + "/" + label + "/" + file)
            elif fold in CV_sets:
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        label = class_label[class_id]
                        copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                 CV_DIR + "/" + label + "/" + file)
            elif fold < 6:
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        label = class_label[class_id]
                        copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                 TRAIN_DIR + "/" + label + "/" + file)
            else:
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = int(file.split("-")[1])
                        label = class_label[class_id]
                        copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                 TEST_DIR + "/" + label + "/" + file)
        CV_sets = CV_sets + 1
    
    timestamp()
    print("Five Fold Dataset Import Succeeded.")

def timestamp():
    result_file = "./ResNeSt_result.txt"
    output_file = "./Output_result.txt"
    target_file = "./Target_result.txt"
    
    recording = open(result_file, 'a')
    output_recording = open(output_file, 'a')
    target_recording = open(target_file, 'a')
    recording.write("===================================================\n")
    output_recording.write("===================================================\n")
    target_recording.write("===================================================\n")
    recording.write(time.strftime("%b %d %Y %H:%M:%S", time.localtime()) + "\n")
    output_recording.write(time.strftime("%b %d %Y %H:%M:%S", time.localtime()) + "\n")
    target_recording.write(time.strftime("%b %d %Y %H:%M:%S", time.localtime()) + "\n")
    recording.close()
    output_recording.close()
    target_recording.close()