import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from STResNeSt.encoding.nn import LabelSmoothing, NLLMultiLabelSmooth
import STResNeSt.encoding as encoding
from STResNeSt.encoding.utils import (accuracy, AverageMeter, MixUpWrapperCPU, LR_Scheduler)

validation_set = [9, 10]

# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []

# Argument
class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='Deep Encoding')
        parser.add_argument('--mini', action='store_true',
                            default=False, help='Load Mini Dataset')
        parser.add_argument('--crop_size', type=int, default=200,
                            help='crop image size (default: 200)')
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

        # Checkpoint Settings
        parser.add_argument('--resume_path', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkpoint_name', type=str, default='default',
                            help='set the checkpoint name')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

args = Options().parse()

# Setting the random seed of PyTorch
torch.manual_seed(10)

# Dataloader
# Load data to memory
train_batch_size = args.train_batch_size
test_batch_size = args.test_batch_size

if args.mini:
    from ResNeSt_Data_Preparation import MiniDataPrepare
    MiniDataPrepare(validation_set)
    train_batch_size = 20
    test_batch_size = 4
else:
    from ResNeSt_Data_Preparation import DataPrepare
    DataPrepare(validation_set)

print("Dataset Loading")
transform_train, transform_val = encoding.transforms.get_transform(
            'imagenet', None, args.crop_size, False)
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

# Checkpoint
if args.resume_path is not None:
    if os.path.isfile(args.resume_path):
        print("=> loading checkpoint '{}'".format(args.resume_path))
        checkpoint = torch.load(args.resume_path)
        args.start_epoch = checkpoint['epoch'] + 1 if args.start_epoch == 0 else args.start_epoch
        best_pred = checkpoint['best_pred']
        acclist_train = checkpoint['acclist_train']
        acclist_val = checkpoint['acclist_val']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimiser.load_state_dict(checkpoint['optimiser'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume_path, checkpoint['epoch']))
    else:
        raise RuntimeError("=> no resume checkpoint found at '{}'". \
                           format(args.resume_path))

# Scheduler
scheduler = LR_Scheduler(args.learning_rate_scheduler,
                             base_lr=args.learning_rate,
                             num_epochs=args.train_epoch,
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=args.warmup_epochs)

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
    if args.resume_path is not None:
        encoding.utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimiser.state_dict(),
            'best_pred': best_pred,
            'acclist_train': acclist_train,
            'acclist_val': acclist_val,
        }, args=args, is_best=is_best)

# Start Training and Validating Process
print("Model Training")
for epoch in range(0, args.train_epoch):
    start = time.time()
    train(epoch)
    validate(epoch)
    elapsed = time.time() - start
    print(f'Epoch: {epoch}, Time cost: {elapsed}')

if args.export:
    torch.save(model.module.state_dict(), args.export + '.pth')
    print("Model Exported")

if args.resume_path is not None:
    encoding.utils.save_checkpoint({
        'epoch': args.epochs-1,
        'state_dict': model.module.state_dict(),
        'optimizer': optimiser.state_dict(),
        'best_pred': best_pred,
        'acclist_train':acclist_train,
        'acclist_val':acclist_val,
        }, args=args, is_best=False)

print(acclist_train)
print(acclist_val)