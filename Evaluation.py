import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, f1_score, auc
import matplotlib.pyplot as plt
from statistics import mean

# Check directory
if not os.path.exists(os.path.expanduser("./PR")):
    os.makedirs(os.path.expanduser("./PR"))
if not os.path.exists(os.path.expanduser("./Accuracy")):
    os.makedirs(os.path.expanduser("./Accuracy"))
if not os.path.exists(os.path.expanduser("./ROC")):
    os.makedirs(os.path.expanduser("./ROC"))
    
Output_result = "Output_result.txt"
Target_result = "Target_result.txt"
output_file = open(Output_result, 'r')
target_file = open(Target_result, 'r')

round_label = ["round1", "round2", "round3", "round4", "round5"]
Model_label = ["ResNeSt50", "ResNeSt101"]
#"ResNeSt200", "ResNeSt269"
Dataset_label = ["Colour_Large_MFCC", "Colour_Large_STFT"]
Dataset_type = ["MFCC", "STFT"]
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

# Scanning prediction and probability
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

# Scanning target
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

# Forming accuracy list for analysis
accuracy_dict = {}

highest_acc_dict = {}

highest_acc_val = 0
best_epoch_val = 0
    
for model in range(len(Model_label)):
    for dataset in range(len(Dataset_label)):
        accuracy_list = np.zeros((3,5,train_epoch))
        for r in range(len(round_label)):
            for epoch in range(train_epoch):
                # Train
                target_list = target_dict["Target_Train" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][epoch]
                predict_list = prediction_dict["Predict_Train" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][epoch]
                target_list = np.array(target_list, dtype='int')
                predict_list = np.array(predict_list, dtype='int')
                classification_metrics = classification_report(target_list, predict_list,target_names = ['engine_idling', 'drilling'],output_dict= True)
                accuracy = classification_metrics['accuracy']
                accuracy_list[0,r,epoch] = accuracy
                    
                # Test
                target_list = target_dict["Target_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][epoch]
                predict_list = prediction_dict["Predict_test_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][epoch]
                target_list = np.array(target_list, dtype='int')
                predict_list = np.array(predict_list, dtype='int')
                classification_metrics = classification_report(target_list, predict_list,target_names = ['engine_idling', 'drilling'],output_dict= True)
                accuracy = classification_metrics['accuracy']
                accuracy_list[1,r,epoch] = accuracy
                    
                # Validation
                target_list = target_dict["Target_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][epoch]
                predict_list = prediction_dict["Predict_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][epoch]
                target_list = np.array(target_list, dtype='int')
                predict_list = np.array(predict_list, dtype='int')
                classification_metrics = classification_report(target_list, predict_list,target_names = ['engine_idling', 'drilling'],output_dict= True)
                accuracy = classification_metrics['accuracy']
                accuracy_list[2,r,epoch] = accuracy
                if highest_acc_val < accuracy:
                    highest_acc_val = accuracy
                    best_epoch_val = epoch
            
            
            highest_acc_dict[Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]] = epoch
            
            highest_acc_val = 0
            best_epoch_val = 0
        accuracy_dict[Model_label[model].lower() + "_" + Dataset_label[dataset]] = accuracy_list
# AUROC, PRROC, F1, sensitivity, specificity
for model in range(len(Model_label)):
    for dataset in range(len(Dataset_label)):
        sensitivity_list = []
        specificity_list = []
        AUROC_list = []
        AUPRC_list = []
        F1_list = []
        cv_test_acc_list = []
        print("Model:", Model_label[model], "Dataset:", Dataset_type[dataset])
        for r in range(len(round_label)):
            best_epoch = highest_acc_dict[Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]]
            
            target_list = target_dict["Target_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][best_epoch]
            predict_list = prediction_dict["Predict_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][best_epoch]
            score_list = score_dict["Score_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][best_epoch]
            target_list = np.array(target_list, dtype='int')
            predict_list = np.array(predict_list, dtype='int')
            score_list = np.array(score_list, dtype='float')
            
            classification_metrics = classification_report(target_list, predict_list, target_names = ['engine_idling', 'drilling'],output_dict= True)
            
            accuracy = classification_metrics['accuracy']
            sensitivity = classification_metrics['drilling']['recall']
            specificity = classification_metrics['engine_idling']['recall']
            f1 = f1_score(target_list, predict_list)
            roc_score = roc_auc_score(target_list, score_list)
            conf_matrix = confusion_matrix(target_list, predict_list)
            precision, recall, thresholds = precision_recall_curve(target_list, score_list)
            pr_score = auc(recall, precision)
            
            print(conf_matrix)
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)
            AUROC_list.append(roc_score)
            AUPRC_list.append(pr_score)
            F1_list.append(f1)
            acc_list = accuracy_dict[Model_label[model].lower() + "_" + Dataset_label[dataset]]
            cv_test_acc_list.append(acc_list[1,r,best_epoch])
        print("Sensitivity:", mean(sensitivity_list))
        print("Specificity:", mean(specificity_list))
        print("AUROC:", mean(AUROC_list))
        print("AUPRC:", mean(AUPRC_list))
        print("F1:", mean(F1_list))
        print(cv_test_acc_list)
        print("CV Testing:", mean(cv_test_acc_list))
            
# Accuracy Curve
for model in range(len(Model_label)):
    for dataset in range(len(Dataset_label)):
        for r in range(len(round_label)):
            plt.figure(figsize=(15,5))
            acc_list = accuracy_dict[Model_label[model].lower() + "_" + Dataset_label[dataset]]
            plt.plot(epoch_label, acc_list[0,r], label = "Train")
            plt.plot(epoch_label, acc_list[1,r], label = "Test")
            plt.plot(epoch_label, acc_list[2,r], label = "Validation")
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel('Accuracy', fontsize=15)
            plt.title("Training accuracy of " + Model_label[model] + " inputting " + Dataset_type[dataset] + " Specturm " + round_label[r], fontsize=18)
            plt.legend(loc='best')
            plt.savefig("./Accuracy/"+Model_label[model]+"_"+Dataset_type[dataset]+"_"+round_label[r])
            plt.clf()


# PR Curve and ROC Curve
for model in range(len(Model_label)):
    for dataset in range(len(Dataset_label)):
        for r in range(len(round_label)):
            best_epoch = highest_acc_dict[Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]]
            
            score = score_dict["Score_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][best_epoch]
            target = target_dict["Target_val_loader" + "_" + Model_label[model].lower() + "_" + Dataset_label[dataset] + "_" + round_label[r]][best_epoch]
            
            score = np.array(score, dtype='float')
            target = np.array(target, dtype='int')
            
            # plot the pr curve
            precision, recall, thresholds = precision_recall_curve(target, score)
            pr_score = auc(recall, precision)
            plt.figure(figsize=(9,9))
            plt.plot(recall,precision, label = "Area under ROC = {:.4f}".format(pr_score))
            plt.title("Precision-Recall Curve of " + Model_label[model] + " inputting " + Dataset_type[dataset] + " Specturm Round " + round_label[r], fontsize=13)
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.savefig("./PR/"+Model_label[model]+"_"+Dataset_type[dataset]+"_"+round_label[r]+".png")
            plt.clf()
            
            # plot the roc curve
            roc_score = roc_auc_score(target, score)
            fpr, tpr, _ = roc_curve(target, score)
            plt.figure(figsize=(9,9))
            plt.plot(fpr, tpr, label = "Area under ROC = {:.4f}".format(roc_score))
            plt.title("ROC Curve of " + Model_label[model] + " inputting " + Dataset_type[dataset] + " Specturm Round " + round_label[r], fontsize=13)
            plt.plot([(0,0),(1,1)],"r-")
            plt.legend(loc = 'best')
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.savefig("./ROC/ROC+" + Model_label[model] + "_" + Dataset_type[dataset] + "Round_" + round_label[r] + ".png")
            plt.clf()
