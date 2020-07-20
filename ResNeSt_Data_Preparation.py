import os
from shutil import copyfile

def DataPrepare(validationFolds, testingFolds):
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

    # Load dataset in class subfolders
    if os.path.exists(DATASET_DIR):
        print("Datasets prepared. Skip preparation step.")
    else:
        print("Standard Dataset Importing.")
        # Create Directories
        os.makedirs(DATASET_DIR + "/urbansound8k/train")
        os.makedirs(DATASET_DIR + "/urbansound8k/val")
        os.makedirs(DATASET_DIR + "/urbansound8k/test/")
        for l in class_label:
            os.makedirs(DATASET_DIR + "/urbansound8k/train/" + l)
            os.makedirs(DATASET_DIR + "/urbansound8k/val/" + l)
            os.makedirs(DATASET_DIR + "/urbansound8k/test/" + l)
        # Classifying and split into train and test set

        for fold in range(1, 11):
            fileList = os.listdir(DOWNLOAD_DIR + '/fold' + str(fold))
            if fold in validationFolds:
                print("Validation set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = file.split("-")[1]
                        label = class_label[int(class_id)]
                        copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                 DATASET_DIR + "/urbansound8k/val/" + label + "/" + file)
                print("Validation set imported")
            elif fold in testingFolds:
                print("Test set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = file.split("-")[1]
                        label = class_label[int(class_id)]
                        copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                 DATASET_DIR + "/urbansound8k/test/" + label + "/" + file)
                print("Test set imported")
            else:
                print("Train set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = file.split("-")[1]
                        label = class_label[int(class_id)]
                        copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                 DATASET_DIR + "/urbansound8k/train/" + label + "/" + file)
                print("Train set imported")
    print("Standard Dataset Import Succeeded.")

# Mini Dataset version is used for reducing training time
# Each class will have at most 100 sample in training set and 10 samples in test set
def MiniDataPrepare(validationFolds, testingFolds):
    DOWNLOAD_DIR = os.path.expanduser("./MFCC")
    DATASET_DIR = os.path.expanduser("~/encoding/data")
    train_set_class_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    test_set_class_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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

    # Load dataset in class subfolders
    if os.path.exists(DATASET_DIR):
        print("Datasets prepared. Skip preparation step.")
    else:
        print("Mini Dataset Importing.")
        # Create Directories
        os.makedirs(DATASET_DIR + "/urbansound8k/train")
        os.makedirs(DATASET_DIR + "/urbansound8k/val")
        for l in class_label:
            os.makedirs(DATASET_DIR + "/urbansound8k/train/" + l)
            os.makedirs(DATASET_DIR + "/urbansound8k/val/" + l)
        # Classifying and split into train and test set
        for fold in range(1, 11):
            fileList = os.listdir(DOWNLOAD_DIR + '/fold' + str(fold))
            if fold in validationFolds:
                print("Validation set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = file.split("-")[1]
                        if not test_set_class_count[int(class_id)] == 10:
                            label = class_label[int(class_id)]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     DATASET_DIR + "/urbansound8k/val/" + label + "/" + file)
                            test_set_class_count[int(class_id)] += 1
                print("Validation set imported")
            elif fold in testingFolds:
                print("Test set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = file.split("-")[1]
                        label = class_label[int(class_id)]
                        copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                 DATASET_DIR + "/urbansound8k/test/" + label + "/" + file)
                print("Test set imported")
            else:
                print("Train set importing...")
                for file in fileList:
                    if not file.startswith('.'):
                        class_id = file.split("-")[1]
                        if not train_set_class_count[int(class_id)] == 100:
                            label = class_label[int(class_id)]
                            copyfile(DOWNLOAD_DIR + '/fold' + str(fold) + "/" + file,
                                     DATASET_DIR + "/urbansound8k/train/" + label + "/" + file)
                            train_set_class_count[int(class_id)] += 1
                print("Train set imported")

    print("Mini Dataset Import Succeeded.")