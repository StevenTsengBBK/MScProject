# MScProject

## Pre-processing
The preprocessing code is in Statistic.ipynb.

urbansound8k_features.csv is the features extracted from the all possible extraction methods provided by librosa package.

Graphs like MFCC, STFT and Waveform are stored in corresponding folders and splited into 10 folders that was organised by UrbanSound8K author.

## Data Prepare for model training

The original dataset is now temporarily split into 2 folders traing and test dataset. The first 9 folds are traing set and last folder is test set.

#### Statistic.py
Is the file that copy file from downloaded folder to home directory named encoding/data/urbansound8k