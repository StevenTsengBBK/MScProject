import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, f1_score, auc
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd

# Check directory
if not os.path.exists(os.path.expanduser("./PR")):
    os.makedirs(os.path.expanduser("./PR"))
if not os.path.exists(os.path.expanduser("./Accuracy")):
    os.makedirs(os.path.expanduser("./Accuracy"))
if not os.path.exists(os.path.expanduser("./ROC")):
    os.makedirs(os.path.expanduser("./ROC"))
if not os.path.exists(os.path.expanduser("./CONF")):
    os.makedirs(os.path.expanduser("./CONF"))

Output_result = "Output_result_holdout.txt"
Target_result = "Target_result_holdout.txt"
output_file = open(Output_result, 'r')
target_file = open(Target_result, 'r')

Model_label = ["ResNeSt50", "ResNeSt101","ResNeSt200", "ResNeSt269"]
Dataset_label = "Colour_Large_STFT"
Dataset_type = "STFT"
Mode_label = ["Train", "holdout_test_loader"]

epoch_label = []
train_epoch = 300

for i in range(train_epoch):
    epoch_label.append(i+1)

combination_list = []
prediction_dict = {}
score_dict = {}
target_dict = {}

def toOneList(l):
    temp = []
    for element in l:
        temp.extend(element)
    return temp


temp_predict = [[],[],[],[]]
count_predict = 0
temp_score = [[],[],[],[]]
count_score = 0
temp_target = [[],[],[],[]]
count_target = 0

# Scanning prediction and probability
for line in output_file:
    key = ""
    if "Predict" in line:
        l = line.split(" | ")
        model = l[5].replace("Model: ", "")
        dataset = l[6].replace("Data: ", "")
        if "Train" in line:
            key = "Predict_Train" + "_" + model + "_" + dataset
        elif "holdout_test_loader" in line:
            key = "Predict_test_loader" + "_" + model + "_" + dataset
        elif "val_loader" in line:
            key = "Predict_val_loader" + "_" + model + "_" + dataset

        if key not in combination_list:
            combination_list.append(key)
            prediction_dict[key] = []
        if l[3] == "GPU: 0":
            temp_predict[0] = l[7].replace("[","").replace("]\n","").split(",")
        if l[3] == "GPU: 1":
            temp_predict[1] = l[7].replace("[","").replace("]\n","").split(",")
        if l[3] == "GPU: 2":
            temp_predict[2] = l[7].replace("[","").replace("]\n","").split(",")
        if l[3] == "GPU: 3":
            temp_predict[3] = l[7].replace("[","").replace("]\n","").split(",")
        count_predict += 1
        if count_predict == 4:
            temp_predict = toOneList(temp_predict)
            prediction_dict[key].append(temp_predict)
            temp_predict = [[],[],[],[]]
            count_predict = 0

    if "Score" in line:
        l = line.split(" | ")
        model = l[5].replace("Model: ", "")
        dataset = l[6].replace("Data: ", "")
        if "Train" in line:
            key = "Score_Train" + "_" + model + "_" + dataset
        elif "holdout_test_loader" in line:
            key = "Score_test_loader" + "_" + model + "_" + dataset

        if key not in combination_list:
            combination_list.append(key)
            score_dict[key] = []
        if l[3] == "GPU: 0":
            temp_score[0] = l[7].replace("[","").replace("]\n","").split(",")
        if l[3] == "GPU: 1":
            temp_score[1] = l[7].replace("[","").replace("]\n","").split(",")
        if l[3] == "GPU: 2":
            temp_score[2] = l[7].replace("[","").replace("]\n","").split(",")
        if l[3] == "GPU: 3":
            temp_score[3] = l[7].replace("[","").replace("]\n","").split(",")
        count_score += 1
        if count_score == 4:
            temp_score = toOneList(temp_score)
            score_dict[key].append(temp_score)
            temp_score = [[],[],[],[]]
            count_score = 0

# Scanning target
for line in target_file:
    key = ""
    if "GPU" in line:
        l = line.split(" | ")
        model = l[5].replace("Model: ", "")
        dataset = l[6].replace("Data: ", "")
        if "Train" in line:
            key = "Target_Train" + "_" + model + "_" + dataset
        elif "holdout_test_loader" in line:
            key = "Target_test_loader" + "_" + model + "_" + dataset
        elif "val_loader" in line:
            key = "Target_val_loader" + "_" + model + "_" + dataset

        if key not in combination_list:
            combination_list.append(key)
            target_dict[key] = []
        if l[3] == "GPU: 0":
            temp_target[0] = l[7].replace("[","").replace("]\n","").split(",")
        if l[3] == "GPU: 1":
            temp_target[1] = l[7].replace("[","").replace("]\n","").split(",")
        if l[3] == "GPU: 2":
            temp_target[2] = l[7].replace("[","").replace("]\n","").split(",")
        if l[3] == "GPU: 3":
            temp_target[3] = l[7].replace("[","").replace("]\n","").split(",")
        count_target += 1
        if count_target == 4:
            temp_target = toOneList(temp_target)
            target_dict[key].append(temp_target)
            temp_target = [[],[],[],[]]
            count_target = 0

# Forming accuracy list for analysis
accuracy_dict = {}
highest_acc_dict = {}

highest_acc_val = 0
best_epoch_val = 0

for model in range(len(Model_label)):
    accuracy_list = np.zeros((3,train_epoch))
    for epoch in range(train_epoch):
        # Train
        target_list = target_dict["Target_Train" + "_" + Model_label[model].lower() + "_" + Dataset_label][epoch]
        predict_list = prediction_dict["Predict_Train" + "_" + Model_label[model].lower() + "_" + Dataset_label][epoch]
        target_list = np.array(target_list, dtype='int')
        predict_list = np.array(predict_list, dtype='int')
        classification_metrics = classification_report(target_list, predict_list,target_names = ['drilling','engine_idling'],output_dict= True)
        accuracy = classification_metrics['accuracy']
        accuracy_list[0,epoch] = accuracy

        # Test
        target_list = target_dict["Target_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label][epoch]
        predict_list = prediction_dict["Predict_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label][epoch]
        target_list = np.array(target_list, dtype='int')
        predict_list = np.array(predict_list, dtype='int')
        classification_metrics = classification_report(target_list, predict_list,target_names = ['drilling','engine_idling'],output_dict= True)
        accuracy = classification_metrics['accuracy']
        accuracy_list[1,epoch] = accuracy

        # Validation
        target_list = target_dict["Target_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label][epoch]
        predict_list = prediction_dict["Predict_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label][epoch]
        target_list = np.array(target_list, dtype='int')
        predict_list = np.array(predict_list, dtype='int')
        classification_metrics = classification_report(target_list, predict_list,target_names = ['drilling','engine_idling'],output_dict= True)
        accuracy = classification_metrics['accuracy']
        accuracy_list[2,epoch] = accuracy
        if highest_acc_val < accuracy:
            highest_acc_val = accuracy
            best_epoch_val = epoch

    highest_acc_dict[Model_label[model].lower() + "_" + Dataset_label] = epoch

    highest_acc_val = 0
    best_epoch_val = 0
    accuracy_dict[Model_label[model].lower() + "_" + Dataset_label] = accuracy_list

# Accuracy Curve
for model in range(len(Model_label)):
    plt.figure(figsize=(15,3))
    acc_list = accuracy_dict[Model_label[model].lower() + "_" + Dataset_label]
    plt.plot(epoch_label, acc_list[0], label = "Train")
    plt.plot(epoch_label, acc_list[1], label = "Holdout")
    plt.plot(epoch_label, acc_list[2], label = "Validation ")
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(Model_label[model], fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Accuracy/Holdout_"+Model_label[model]+"_"+Dataset_type)
    plt.clf()

# AUROC, PRROC, F1, sensitivity, specificity
for model in range(len(Model_label)):
    print("Model:", Model_label[model], "Dataset:", Dataset_type)

    best_epoch = highest_acc_dict[Model_label[model].lower() + "_" + Dataset_label]

    target_list = target_dict["Target_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label][best_epoch]
    predict_list = prediction_dict["Predict_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label][best_epoch]
    score_list = score_dict["Score_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label][best_epoch]
    target_list = np.array(target_list, dtype='int')
    predict_list = np.array(predict_list, dtype='int')
    score_list = np.array(score_list, dtype='float')

    classification_metrics = classification_report(target_list, predict_list, target_names = ['drilling','engine_idling'],output_dict= True)

    accuracy = classification_metrics['accuracy']
    sensitivity = classification_metrics['drilling']['recall']
    specificity = classification_metrics['engine_idling']['recall']
    f1 = f1_score(target_list, predict_list)
    roc_score = roc_auc_score(target_list, score_list)
    conf_matrix = confusion_matrix(target_list, predict_list)
    precision, recall, thresholds = precision_recall_curve(target_list, score_list)
    pr_score = auc(recall, precision)

    print(conf_matrix)
    print("Accuracy:", accuracy)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("AUROC:", roc_score)
    print("AUPRC:", pr_score)
    print("F1:", f1)

# PR Curve and ROC Curve
# plot the pr curve
plt.figure(figsize=(4,4))
for model in range(len(Model_label)):
    best_epoch = highest_acc_dict[Model_label[model].lower() + "_" + Dataset_label]

    score = score_dict["Score_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label][best_epoch]
    target = target_dict["Target_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label][best_epoch]

    score = np.array(score, dtype='float')
    target = np.array(target, dtype='int')

    precision, recall, thresholds = precision_recall_curve(target, score)
    pr_score = auc(recall, precision)
    
    plt.plot(recall,precision, label = Model_label[model] + " AUPRC = {:.4f}".format(pr_score))
plt.plot([(0,0),(1,1)],"k--")
plt.title("PR-Curve", fontsize=14)
plt.legend(loc = 'best')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("./PR/Holdout.png")
plt.clf()

 # plot the roc curve
plt.figure(figsize=(4,4))
for model in range(len(Model_label)):
    best_epoch = highest_acc_dict[Model_label[model].lower() + "_" + Dataset_label]

    score = score_dict["Score_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label][best_epoch]
    target = target_dict["Target_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label][best_epoch]

    score = np.array(score, dtype='float')
    target = np.array(target, dtype='int')
    
    roc_score = roc_auc_score(target, score)
    fpr, tpr, _ = roc_curve(target, score)

    plt.plot(fpr, tpr, label = Model_label[model] + " AUROC = {:.4f}".format(roc_score))
plt.title("ROC-Curve", fontsize=14)
plt.plot([(0,0),(1,1)],"r-")
plt.legend(loc = 'best')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("./ROC/Holdout_ROC.png")
plt.clf()
