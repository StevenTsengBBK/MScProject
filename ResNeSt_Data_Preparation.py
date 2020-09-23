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

DATASET_TYPE = ["/Colour_Large_MFCC", "/Colour_Large_STFT"]

DATASET_DIR = os.path.expanduser("./encoding/data")

def DataPrepare(CLASS1_LABELID, CLASS2_LABELID):
    
    # Load dataset in class subfolders
    mflag = False
    sflag = False
    
    if os.path.exists(DATASET_DIR + DATASET_TYPE[0]):
        mflag = True
    if os.path.exists(DATASET_DIR + DATASET_TYPE[0]):
        sflag = True

    if mflag and sflag:
        return
   
    queue = []
    if not mflag:
        queue.append(DATASET_TYPE[0])
    if not sflag:
        queue.append(DATASET_TYPE[1])
    
    # Create Directories
    print("Five Fold Dataset Importing.")

    for dtype in queue:
        DOWNLOAD_DIR = os.path.expanduser("./" + dtype)
        # Load dataset in class subfolders
        CV_sets = np.array([1])
        validation_sets = np.array([6,7])
        
        dataset_type_dir = DATASET_DIR + dtype
        
        os.makedirs(dataset_type_dir +"/urbansound8k/holdoutTrain")
        os.makedirs(dataset_type_dir +"/urbansound8k/holdoutTest")
        os.makedirs(dataset_type_dir +"/urbansound8k/val")
        os.makedirs(dataset_type_dir +"/urbansound8k/holdoutTrain/" + class_label[CLASS1_LABELID])
        os.makedirs(dataset_type_dir +"/urbansound8k/holdoutTest/" + class_label[CLASS1_LABELID])
        os.makedirs(dataset_type_dir +"/urbansound8k/val/" + class_label[CLASS1_LABELID])
        os.makedirs(dataset_type_dir +"/urbansound8k/holdoutTrain/" + class_label[CLASS2_LABELID])
        os.makedirs(dataset_type_dir +"/urbansound8k/holdoutTest/" + class_label[CLASS2_LABELID])
        os.makedirs(dataset_type_dir +"/urbansound8k/val/" + class_label[CLASS2_LABELID])
        HOLDOUT_TRAIN_DIR = dataset_type_dir +"/urbansound8k/holdoutTrain"
        HOLDOUT_TEST_DIR = dataset_type_dir +"/urbansound8k/holdoutTest"
        HOLDOUT_VAL_DIR = dataset_type_dir +"/urbansound8k/val"
        
        for iteration in range(1,6):
            ITER_DIR = dataset_type_dir + "/round" + str(iteration) + "/urbansound8k"
            os.makedirs(ITER_DIR +"/train")
            os.makedirs(ITER_DIR +"/val")
            os.makedirs(ITER_DIR + "/CV")
            os.makedirs(ITER_DIR +"/train/" + class_label[CLASS1_LABELID])
            os.makedirs(ITER_DIR +"/val/" + class_label[CLASS1_LABELID])
            os.makedirs(ITER_DIR + "/CV/" + class_label[CLASS1_LABELID])
            os.makedirs(ITER_DIR +"/train/" + class_label[CLASS2_LABELID])
            os.makedirs(ITER_DIR +"/val/" + class_label[CLASS2_LABELID])
            os.makedirs(ITER_DIR + "/CV/" + class_label[CLASS2_LABELID])

            TRAIN_DIR = ITER_DIR +"/train"
            VAL_DIR = ITER_DIR +"/val"
            
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
                if fold < 6 and iteration == 5:
                    for file in fileList:
                        if not file.startswith('.'):
                            class_id = int(file.split("-")[1])
                            if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                                label = class_label[class_id]
                                copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                         HOLDOUT_TRAIN_DIR + "/" + label + "/" + file)
                if fold <8 and fold >5 and iteration == 5:
                    for file in fileList:
                        if not file.startswith('.'):
                            class_id = int(file.split("-")[1])
                            if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                                label = class_label[class_id]
                                copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                         HOLDOUT_VAL_DIR + "/" + label + "/" + file)
                if fold > 7 and iteration == 5:
                    for file in fileList:
                        if not file.startswith('.'):
                            class_id = int(file.split("-")[1])
                            if class_id == CLASS1_LABELID or class_id == CLASS2_LABELID:
                                label = class_label[class_id]
                                copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                         HOLDOUT_TEST_DIR + "/" + label + "/" + file)
            CV_sets = CV_sets + 1

        timestamp()

        
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