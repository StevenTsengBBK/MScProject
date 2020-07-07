import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

import STResNeSt.encoding as encoding
from encoding.utils import (accuracy, AverageMeter, MixUpWrapper, LR_Scheduler, torch_dist_sum)

validation_set = [9, 10]

# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []
# model including resnest50 resnest101 resnest200 resnest269 resnest50_fast resnest101_fast
model_name = "resnest50"
train_batch_size = 200
test_batch_size = 200
train_epoch = 10
learning_rate = 0.001
momentum = 0.9
weight_decay = 1e-4

# Argument
class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='Deep Encoding')
        parser.add_argument('--mini', action='store_true',
                            default=False, help='Load Mini Dataset')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

args = Options().parse()

if args.mini:
    from ResNeSt_Data_Preparation import MiniDataPrepare
    MiniDataPrepare(validation_set)
    train_batch_size = 20
    test_batch_size = 4
else:
    from ResNeSt_Data_Preparation import DataPrepare
    DataPrepare(validation_set)

# Setting the random seed of PyTorch
torch.manual_seed(10)
torch.cuda.manual_seed(10)

# Dataloader
# Load data to memory
print("Dataset Loading")
transform_train, transform_val = encoding.transforms.get_transform(
            'imagenet', None, 224, False)
trainset = encoding.datasets.get_dataset('imagenet', root=os.path.expanduser('~/encoding/data'),
                                             transform=transform_train, train=True, download=True)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False)


validationset = encoding.datasets.get_dataset('imagenet', root=os.path.expanduser('~/encoding/data'),
                                           transform=transform_val, train=False, download=True)
val_loader = torch.utils.data.DataLoader(validationset, batch_size=test_batch_size, shuffle=False)

print("Dataset Loaded")

# Model Training
# Check if computer has GPU
device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"
    print("GPU device not available, CPU has replaced instead.")
device = torch.device(device)

# Acquire model from model zoo
model = encoding.models.get_model(model_name)
print("Model Acquired.")

# Criterion and Optimiser
# There are 2 other methods
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
print("Loss function and optimiser ready.")

def train(epoch):
    print("Training epoch: ", str(epoch))
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    global best_pred, acclist_train
    for batch_idx, (data, target) in enumerate(train_loader):
        optimiser.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimiser.step()
        acc1 = accuracy(output, target, topk=(1,))
        top1.update(acc1[0], data.size(0))
        losses.update(loss.item(), data.size(0))
        # if (batch_idx+1) % 100 == 0:
        print('Batch: %d| Loss: %.3f | Top1: %.3f' % (batch_idx, losses.avg, top1.avg))
    print(float(top1.avg))
    acclist_train += [float(top1.avg)]
    print("Training epoch: ", str(epoch), "finished")

def validate(epoch):
    print("Validating epoch: ", str(epoch))
    model.eval()
    top1 = AverageMeter()
    global best_pred, acclist_train, acclist_val
    is_best = False
    for batch_idx, (data, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model(data)
            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1[0], data.size(0))
            print('Batch: %d|Top1: %.3f' % (batch_idx, top1.avg))

    acclist_val += [float(top1.avg)]
    print('Validation: Top1: %.3f' % (top1.avg))
    print("Validating epoch: ", str(epoch), "finished")

print("Model Training")
for epoch in range(0, train_epoch):
    start = time.time()
    train(epoch)
    validate(epoch)
    elapsed = time.time() - start
    print(f'Epoch: {epoch}, Time cost: {elapsed}')

print(acclist_train)
print(acclist_val)