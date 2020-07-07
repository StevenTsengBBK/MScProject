# MScProject

## Pre-processing
The preprocessing code is in Statistic.ipynb.

urbansound8k_features.csv is the features extracted from the all possible extraction methods provided by librosa package.

Graphs like MFCC, STFT and Waveform are stored in corresponding folders and splited into 10 folders that was organised by UrbanSound8K author.

## Data Preparing

### ResNeSt_Data_Preparation.py
The code that copy data from download folder to root directory ~/encoding/data. Each class has a folder to collect the samples.

For lite version, use ```--mini``` when training the model. The collection will not acquire 100 samples in each class for training and 10 samples in each class for testing.

## Model Training

### ResNeSt_Model_Train.py

This is the ResNeSt training code with validation process.