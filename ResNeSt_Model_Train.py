##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yao-I Tseng
## Reference: Hang Zhang (2020) https://github.com/zhanghang1989/ResNeSt
## Email: mrsuccess1203@gmail.com
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from ResNeSt_Data_Preparation import *

from STResNeSt.encoding.nn import LabelSmoothing, NLLMultiLabelSmooth
import STResNeSt.encoding as encoding
from STResNeSt.encoding.utils import (accuracy, AverageMeter, MixUpWrapperCPU, LR_Scheduler)

# global variable
best_pred = 0.0
acclist_train = np.array([])
acclist_val = np.array([])
acclist_CVTest = np.array([])
acclist_HoldoutTest = np.array([])

# Argument
class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='Deep Encoding')
        parser.add_argument('--mini', action='store_true',
                            default=False, help='Load Mini Dataset')
        parser.add_argument('--crop_size', type=int, default=271,
                            help='crop image size (default: 271)')
        parser.add_argument('--mixup', type=float, default=0.0,
                            help='mixup (default eta: 0.0)')
        parser.add_argument('--label_smoothing', type=float, default=0.0,
                            help='label-smoothing (default eta: 0.0)')

        # Model Settings
        # model including resnest50 resnest101 resnest200 resnest269 resnest50_fast resnest101_fast
        parser.add_argument('--model_name', type=str, default='resnest50',
                            help='Training model (default: resnest50)')
        parser.add_argument('--rectify', action='store_true',
                            default=False, help='rectify convolution')
        parser.add_argument('--rectify_avg', action='store_true',
                            default=False, help='rectify convolution')
        parser.add_argument('--last_gamma', action='store_true', default=False,
                            help='whether to init gamma of the last BN layer in \
                                    each bottleneck to 0 (default: False)')
        parser.add_argument('--dropblock_prob', type=float, default=0,
                            help='DropBlock prob. default is 0.')
        parser.add_argument('--final_drop', type=float, default=0,
                            help='final dropout prob. default is 0.')
        parser.add_argument('--train_batch_size', type=int, default=100, metavar='N',
                            help='batch size for training (default: 100)')
        parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                            help='batch size for testing (default: 100)')
        parser.add_argument('--train_epoch', type=int, default=100, metavar='N',
                            help='number of epochs to train (default: 100)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='the epoch number to start (default: 0)')
        parser.add_argument('--FiveFold', action='store_true', default=False, help="5fold Validation")
        parser.add_argument('--TenFold', action='store_true', default=False, help="10fold Validation")

        # Optimiser Settings
        parser.add_argument('--learning_rate', type=float, default=0.1, metavar='LR',
                            help='learning rate (default: 0.1)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            metavar='M', help='SGD weight decay (default: 1e-4)')
        parser.add_argument('--warmup_epochs', type=int, default=0,
                            help='number of warmup epochs (default: 0)')
        parser.add_argument('--learning_rate_scheduler', type=str, default='cos',
                            help='learning rate scheduler (default: cos)')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

def model_prepare(root = '~/encoding/data'):
    print("Dataset Loading")
    transform_train, transform_val = encoding.transforms.get_transform(
                'imagenet', None, args.crop_size, False)
    trainset = encoding.datasets.get_dataset('imagenet', root=os.path.expanduser(root),
                                                 transform=transform_train, train=True, download=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=False)

    validationset = encoding.datasets.get_dataset('imagenet', root=os.path.expanduser(root),
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

    # Initialising model
    # Argument acquiring
    model_kwargs = {}
    if args.final_drop > 0.0:
        model_kwargs['final_drop'] = args.final_drop

    if args.dropblock_prob > 0.0:
        model_kwargs['dropblock_prob'] = args.dropblock_prob

    if args.last_gamma:
        model_kwargs['last_gamma'] = True

    if args.rectify:
        model_kwargs['rectified_conv'] = True
        model_kwargs['rectify_avg'] = args.rectify_avg

    # Acquire model from model zoo
    model = encoding.models.get_model(args.model_name, **model_kwargs)
    print("Model Acquired.")

    if args.dropblock_prob > 0.0:
        from functools import partial
        from encoding.nn import reset_dropblock

        nr_iters = (args.train_epoch - args.warmup_epochs) * len(train_loader)
        apply_drop_prob = partial(reset_dropblock, args.warmup_epochs * len(train_loader),
                                  nr_iters, 0.0, args.dropblock_prob)
        model.apply(apply_drop_prob)

    # Criterion and Optimiser
    if args.mixup > 0:
        train_loader = MixUpWrapperCPU(args.mixup, 1000, train_loader)
        criterion = NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        criterion = LabelSmoothing(args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    print("Loss function and optimiser ready.")

    # Scheduler
    scheduler = LR_Scheduler(args.learning_rate_scheduler,
                                 base_lr=args.learning_rate,
                                 num_epochs=args.train_epoch,
                                 iters_per_epoch=len(train_loader),
                                 warmup_epochs=args.warmup_epochs)

    return model, train_loader, val_loader, scheduler, optimiser, criterion

def train(epoch):
    print("Training epoch: ", str(epoch))
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    global best_pred, acclist_train
    for batch_idx, (data, target) in enumerate(train_loader):
        scheduler(optimiser, batch_idx, epoch, best_pred)
        optimiser.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimiser.step()

        # The expanded size of the tensor (20) must match the existing size (20000)
        if not args.mixup:
            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1[0], data.size(0))

        losses.update(loss.item(), data.size(0))
        if args.mixup:
            print('Batch: %d| Loss: %.3f' % (batch_idx, losses.avg))
        else:
            print('Batch: %d| Loss: %.3f | Top1 Accuracy: %.3f' % (batch_idx, losses.avg, top1.avg))
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
    print('Validation: Top1 Accuracy: %.3f' % (top1.avg))
    print("Validating epoch: ", str(epoch), "finished")

    if top1.avg > best_pred:
        best_pred = top1.avg
        is_best = True

def test(root = '~/encoding/data', CV=False):
    print("Start Testing Process")

    # Initialise Data Loader
    _, transform_test = encoding.transforms.get_transform(
        'imagenet', None, args.crop_size, False)
    testset = encoding.datasets.get_dataset('imagenet', root=os.path.expanduser(root),
                                             transform=transform_test, train=False, test=True, download=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    # Model Testing
    model.eval()
    top1 = AverageMeter()
    global acclist_train, acclist_CVTest, acclist_HoldoutTest
    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            output = model(data)
            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1[0], data.size(0))
            print('Batch: %d|Top1: %.3f' % (batch_idx, top1.avg))

    if CV:
        acclist_CVTest += [float(top1.avg)]
        print("Cross Validation Test:", top1.avg)
    else:
        acclist_HoldoutTest += [float(top1.avg)]
        print("Holdout Test:", top1.avg)
    print("Testing Done")

validation_set = [8, 9]
testing_set = [10]

# Loading arguments
args = Options().parse()

# Setting the random seed of PyTorch
torch.manual_seed(10)

# Data loader
# Load data to memory
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size

# The mini train allow to use small dataset (at most 100 images in both train and validation sets)
# Otherwise the the system will prepare full dataset or for 5fold validation specific dataset
# The mini train and standard train have fixed validation set (8th and 9th folds) and testing set (10th fold)
if args.mini:
    if args.FiveFold:
        raise Exception("Mini Train Does not Allow Five Fold.")
    else:
        MiniDataPrepare()
    train_batch_size = 20
    test_batch_size = 4
else:
    if args.FiveFold:
        DataPrepareFiveFold()
    else:
        DataPrepare()

# Start Training and Validating Process
print("Model Training")
if args.FiveFold:
    # Five Fold Validation
    print("5-Fold Validation activated")
    validation_set = np.array([1])
    for round in range(1,6):
        print("Starting round", str(round), "Validation Set:", str(validation_set))

        # Setting the root to fetch the dataset
        root = '~/encoding/data/round' + str(round)

        # Preparing model
        model, train_loader, val_loader, scheduler, optimiser, criterion = model_prepare(root)
        for epoch in range(0, args.train_epoch):
            start = time.time()
            train(epoch)
            # Using Fold_6 and Fold_7 for validation
            validate(epoch)
            elapsed = time.time() - start
            print(f'Epoch: {epoch}, Time cost: {elapsed}')
        # Using one of the five fold for Cross Validation Test
        test(root, CV=True)
        # Using Holdout Test Set for Final Testing
        test(root)
        validation_set = validation_set + 1
    print("Cross Validation Test:", acclist_CVTest)
    print("Holdout Test:", acclist_HoldoutTest)
    print("Cross Validation Result: ", np.mean(acclist_val))
    print("Holdout Test Result:", np.mean(acclist_HoldoutTest))


else:
    # Standard Training, Validation, and Testing Model
    # Preparing model
    model, train_loader, val_loader, scheduler, optimiser, criterion = model_prepare()
    for epoch in range(0, args.train_epoch):
        start = time.time()
        train(epoch)
        validate(epoch)
        elapsed = time.time() - start
        print(f'Epoch: {epoch}, Time cost: {elapsed}')
    # Using one of the five fold for Cross Validation Test
    test(CV=True)
    # Using Holdout Test Set for Final Testing
    test()
    print("Training Result: ", acclist_train)
    print("Validation Result: ", acclist_val)
    print("Cross Validation Test:", acclist_CVTest)
    print("Holdout Test:", acclist_HoldoutTest)

if args.export:
    torch.save(model.module.state_dict(), args.export + '.pth')
    print("Model Exported")

print("Training Result: ", acclist_train)
print("Validation Result: ", acclist_val)