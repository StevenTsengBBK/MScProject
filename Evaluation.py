import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, f1_score
import matplotlib.pyplot as plt

# Check directory

if not os.path.exists(os.path.expanduser("./PR")):
    os.makedirs(os.path.expanduser("./PR"))
if not os.path.exists(os.path.expanduser("./Accuracy")):
    os.makedirs(os.path.expanduser("./Accuracy"))
    
Output_result = "Output_result.txt"
Target_result = "Target_result.txt"
output_file = open(Output_result, 'r')
target_file = open(Target_result, 'r')

round_label = ["round1", "round2", "round3", "round4", "round5"]
Model_label = ["ResNeSt50"]
#, "ResNeSt101", "ResNeSt200", "ResNeSt269"
Dataset_label = ["Colour_Large_MFCC"]
# , "Colour_Large_STFT"
Mode_label = ["Train", "test_loader", "val_loader"]
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
for line in output_file:
    key = ""
    if "Predict" in line:
        l = line.split(" | ")
        model = l[5].replace("Model: ", "")
        dataset = l[6].replace("Data: ", "")
        cvRound = l[1].replace("Round: ", "")
        if "Train" in line:
            key = "Predict_Train" + "_" + model + "_" + dataset + "_round" + cvRound
        elif "test_loader" in line:
            key = "Predict_test_loader" + "_" + model + "_" + dataset + "_round" + cvRound
        elif "val_loader" in line:
            key = "Predict_val_loader" + "_" + model + "_" + dataset + "_round" + cvRound
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
        cvRound = l[1].replace("Round: ", "")
        if "Train" in line:
            key = "Score_Train" + "_" + model + "_" + dataset + "_round" + cvRound
        elif "test_loader" in line:
            key = "Score_test_loader" + "_" + model + "_" + dataset + "_round" + cvRound
        elif "val_loader" in line:
            key = "Score_val_loader" + "_" + model + "_" + dataset + "_round" + cvRound
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

temp_target = [[],[],[],[]]
count_target = 0
for line in target_file:
    key = ""
    if "GPU" in line:
        l = line.split(" | ")
        model = l[5].replace("Model: ", "")
        dataset = l[6].replace("Data: ", "")
        cvRound = l[1].replace("Round: ", "")
        if "Train" in line:
            key = "Target_Train" + "_" + model + "_" + dataset + "_round" + cvRound
        elif "test_loader" in line:
            key = "Target_test_loader" + "_" + model + "_" + dataset + "_round" + cvRound
        elif "val_loader" in line:
            key = "Target_val_loader" + "_" + model + "_" + dataset + "_round" + cvRound
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
            
accuracy_list = np.zeros((2,3,5,train_epoch))

highest_acc_train = 0
best_epoch_train = 0
highest_acc_test = 0
best_epoch_test = 0
highest_acc_val = 0
best_epoch_val = 0
    
for model in range(len(Model_label)):
    for dataset in range(len(Dataset_label)):
        for r in range(len(round_label)):
            for epoch in range(train_epoch):
                # Train
                target_list = target_dict["Target_Train" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][epoch]
                predict_list = prediction_dict["Predict_Train" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][epoch]
                target_list = np.array(target_list, dtype='int')
                predict_list = np.array(predict_list, dtype='int')
                classification_metrics = classification_report(target_list, predict_list,target_names = ['engine_idling', 'drilling'],output_dict= True)
                accuracy = classification_metrics['accuracy']
                accuracy_list[dataset,0,r,epoch] = accuracy
                if highest_acc_train < accuracy:
                    highest_acc_train = accuracy
                    best_epoch_train = epoch
                if epoch == train_epoch - 1:
                    sensitivity = classification_metrics['drilling']['recall']
                    specificity = classification_metrics['engine_idling']['recall']
                    f1 = f1_score(target_list, predict_list)
                    conf_matrix = confusion_matrix(target_list, predict_list)
                    print("sensitivity", sensitivity, "specificity", specificity, "f1", f1)
                    print(conf_matrix)
                # Test
                target_list = target_dict["Target_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][epoch]
                predict_list = prediction_dict["Predict_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][epoch]
                target_list = np.array(target_list, dtype='int')
                predict_list = np.array(predict_list, dtype='int')
                classification_metrics = classification_report(target_list, predict_list,target_names = ['engine_idling', 'drilling'],output_dict= True)
                accuracy = classification_metrics['accuracy']
                accuracy_list[dataset,1,r,epoch] = accuracy
                if highest_acc_test < accuracy:
                    highest_acc_test = accuracy
                    best_epoch_test = epoch
                if epoch == train_epoch - 1:
                    sensitivity = classification_metrics['drilling']['recall']
                    specificity = classification_metrics['engine_idling']['recall']
                    f1 = f1_score(target_list, predict_list)
                    conf_matrix = confusion_matrix(target_list, predict_list)
                    print("sensitivity", sensitivity, "specificity", specificity, "f1", f1)
                    print(conf_matrix)
                # Validation
                target_list = target_dict["Target_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][epoch]
                predict_list = prediction_dict["Predict_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][epoch]
                target_list = np.array(target_list, dtype='int')
                predict_list = np.array(predict_list, dtype='int')
                classification_metrics = classification_report(target_list, predict_list,target_names = ['engine_idling', 'drilling'],output_dict= True)
                accuracy = classification_metrics['accuracy']
                accuracy_list[dataset,2,r,epoch] = accuracy
                if highest_acc_val < accuracy:
                    highest_acc_val = accuracy
                    best_epoch_val = epoch
                if epoch == train_epoch -1:
                    sensitivity = classification_metrics['drilling']['recall']
                    specificity = classification_metrics['engine_idling']['recall']
                    f1 = f1_score(target_list, predict_list)
                    conf_matrix = confusion_matrix(target_list, predict_list)
                    print("sensitivity", sensitivity, "specificity", specificity, "f1", f1)
                    print(conf_matrix)
            print("highest_acc_train", highest_acc_train, "best_epoch_train", best_epoch_train, "highest_acc_test", highest_acc_test, "best_epoch_test", best_epoch_test, "highest_acc_val", highest_acc_val, "best_epoch_val", best_epoch_val)
            highest_acc_train = 0
            best_epoch_train = 0
            highest_acc_test = 0
            best_epoch_test = 0
            highest_acc_val = 0
            best_epoch_val = 0

# Accuracy Curve
for dataset in range(len(Dataset_label)):
    for r in range(len(round_label)):
        plt.figure(figsize=(15,5))
        plt.plot(epoch_label, accuracy_list[dataset,0,r], label = "Train")
        plt.plot(epoch_label, accuracy_list[dataset,1,r], label = "Test")
        plt.plot(epoch_label, accuracy_list[dataset,2,r], label = "Validation")
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.title("Training accuracy of " + Model_label[0] + " inputting " + Dataset_label[dataset] + " Specturm " + round_label[r], fontsize=18)
        plt.legend(loc='best')
        plt.savefig("./Accuracy/"+Model_label[0]+"_"+Dataset_label[dataset]+"_"+round_label[r])
        plt.clf()


# PR Curve
for model in range(len(Model_label)):
    for dataset in range(len(Dataset_label)):
        for r in range(len(round_label)):
            score = score_dict["Score_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][99]
            target = target_dict["Target_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][99]
            score = np.array(score, dtype='float')
            target = np.array(target, dtype='int')
            precision, recall, thresholds = precision_recall_curve(target, score)
            plt.figure(figsize=(9,9))
            plt.plot(recall,precision)
            plt.title("Precision-Recall Curve of " + Model_label[0] + " inputting " + Dataset_label[dataset] + " Specturm " + round_label[r], fontsize=13)
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.savefig("./PR/"+Model_label[0]+"_"+Dataset_label[dataset]+"_"+round_label[r])
            plt.clf()