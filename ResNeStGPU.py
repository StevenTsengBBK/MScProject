##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Referenced by: Hang Zhang https://github.com/zhanghang1989/PyTorch-Encoding/
## Created by: Yao-I Tseng
## Email: mrsuccess1203@gmail.com
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from ResNeSt import encoding
from ResNeSt.encoding.nn import LabelSmoothing, NLLMultiLabelSmooth
from ResNeSt.encoding.utils import (accuracy, AverageMeter, MixUpWrapper, LR_Scheduler, torch_dist_sum)

from ResNeSt_Data_Preparation import *

class Options():
    def __init__(self):
        # data settings
        parser = argparse.ArgumentParser(description='Deep Encoding')
        parser.add_argument('--dataset', type=str, default='imagenet',
                            help='training dataset (default: imagenet)')
        parser.add_argument('--base-size', type=int, default=None,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=271,
                            help='crop image size')
        parser.add_argument('--label-smoothing', type=float, default=0.0,
                            help='label-smoothing (default eta: 0.0)')
        parser.add_argument('--mixup', type=float, default=0.0,
                            help='mixup (default eta: 0.0)')
        parser.add_argument('--rand-aug', action='store_true',
                            default=False, help='random augment')
        parser.add_argument('--CLASS1_LABELID', type=int, default=4, metavar='N',
                            help='Label 1')
        parser.add_argument('--CLASS2_LABELID', type=int, default=5, metavar='N',
                            help='Label 2')
        parser.add_argument('--Download_folder', type=str, default='Colour_Large_MFCC',
                            help='Download folder')
        # model params
        parser.add_argument('--model', type=str, default='densenet',
                            help='network model type (default: densenet)')
        parser.add_argument('--rectify', action='store_true',
                            default=False, help='rectify convolution')
        parser.add_argument('--rectify-avg', action='store_true',
                            default=False, help='rectify convolution')
        parser.add_argument('--pretrained', action='store_true',
                            default=False, help='load pretrianed mode')
        parser.add_argument('--last-gamma', action='store_true', default=False,
                            help='whether to init gamma of the last BN layer in \
                            each bottleneck to 0 (default: False)')
        parser.add_argument('--dropblock-prob', type=float, default=0,
                            help='DropBlock prob. default is 0.')
        parser.add_argument('--final-drop', type=float, default=0,
                            help='final dropout prob. default is 0.')
        parser.add_argument('--class-num', type=float, default=2,
                            help='Number of classes.')
        # Testing params
        parser.add_argument('--FiveFold', action='store_true', default=False, help="5fold Validation")

        # training params
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='batch size for training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                            help='batch size for testing (default: 256)')
        parser.add_argument('--epochs', type=int, default=120, metavar='N',
                            help='number of epochs to train (default: 600)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='the epoch number to start (default: 1)')
        parser.add_argument('--workers', type=int, default=8,
                            metavar='N', help='dataloader threads')
        # optimizer
        parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                            help='learning rate (default: 0.1)')
        parser.add_argument('--lr-scheduler', type=str, default='cos',
                            help='learning rate scheduler (default: cos)')
        parser.add_argument('--warmup-epochs', type=int, default=0,
                            help='number of warmup epochs (default: 0)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar ='M', help='SGD weight decay (default: 1e-4)')
        parser.add_argument('--no-bn-wd', action='store_true',
                            default=False, help='no bias decay')
        # seed
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        # distributed
        parser.add_argument('--world-size', default=1, type=int,
                            help='number of nodes for distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='node rank for distributed training')
        parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                            help='url used to set up distributed training')
        parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
        # evaluation option
        parser.add_argument('--eval', action='store_true', default= False,
                            help='evaluating')
        parser.add_argument('--export', type=str, default=None,
                            help='put the path to resuming file if needed')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args

# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []
cv_acclist_val = []
acclist_train_set = []
acclist_val_set = []
result_file = "./ResNeSt_result.txt"
output_file = "./Output_result.txt"
target_file = "./Target_result.txt"

def main():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Using device:', dev)
    print("GPU Devices:", torch.cuda.device_count())
    print("Programme Start")
    args = Options().parse()

    print("Data preparing")
    if args.FiveFold:
        DataPrepareFiveFold(args.CLASS1_LABELID, args.CLASS2_LABELID, args.Download_folder)
    else:
        DataPrepare(args.CLASS1_LABELID, args.CLASS2_LABELID, args.Download_folder)
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size
    args.lr = args.lr * args.world_size
    
    args.kfold = 1
    print("Training Start")

    if args.FiveFold:
        train_round = 6
    else:
        train_round = 1

    recording = open(result_file, 'a')
    output_recording = open(output_file, 'a')
    target_recording = open(target_file, 'a')
    recording.write("Arguments\n" + str(args) + "\n")
    output_recording.write("Arguments\n" + str(args) + "\n")
    target_recording.write("Arguments\n" + str(args) + "\n")
       
    for f in range(1,train_round):
        args.kfold = f
        print("Round", args.kfold)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    args.rank = args.rank * ngpus_per_node + gpu
    print('rank: {} / {}'.format(args.rank, args.world_size))
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    torch.cuda.set_device(args.gpu)
    # init the args
    global best_pred, acclist_train, acclist_val

    if args.gpu == 0:
        print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # init dataloader
    transform_train, transform_val = encoding.transforms.get_transform(
            args.dataset, args.base_size, args.crop_size, args.rand_aug)

    validation_loader = {}
    if args.FiveFold:
        print("Loading Fold", args.kfold)
        trainset = encoding.datasets.get_dataset(args.dataset, root=os.path.expanduser('./encoding/data/round'+str(args.kfold)),
                                             transform=transform_train, train=True, download=True)
        cv_valset = encoding.datasets.get_dataset(args.dataset, root=os.path.expanduser('./encoding/data/round'+str(args.kfold)),
                                           transform=transform_val, train=False, cv_val=True, download=True)
        general_valset = encoding.datasets.get_dataset(args.dataset, root=os.path.expanduser('./encoding/data/round'+str(args.kfold)),
                                           transform=transform_val, train=False, cv_val=False, hold_test=False, download=True)
        holdout_testset = encoding.datasets.get_dataset(args.dataset,
                                                        root=os.path.expanduser('./encoding/data/round'+str(args.kfold)),
                                           transform=transform_val, train=False, cv_val=False, hold_test=True, download=True)

        # Training Set Fold [1,2,3,4,5] depends on the fold
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            sampler=train_sampler)

        # Five fold valudation
        val_sampler = torch.utils.data.distributed.DistributedSampler(cv_valset, shuffle=False)
        val_loader = torch.utils.data.DataLoader(
            cv_valset, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            sampler=val_sampler)

        # Validation for overfitting test
        general_val_sampler = torch.utils.data.distributed.DistributedSampler(general_valset, shuffle=False)
        general_val_loader = torch.utils.data.DataLoader(
            general_valset, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            sampler=general_val_sampler)

        # Validation for holdout
        holdout_test_sampler = torch.utils.data.distributed.DistributedSampler(holdout_testset, shuffle=False)
        holdout_test_loader = torch.utils.data.DataLoader(
            holdout_testset, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            sampler=holdout_test_sampler)

        validation_loader = {"val_loader":val_loader,
                             "general_val_loader":general_val_loader,
                             "holdout_test_loader":holdout_test_loader}

    else:
        trainset = encoding.datasets.get_dataset(args.dataset, root=os.path.expanduser('./encoding/data'),
                                             transform=transform_train, train=True, download=True)
        valset = encoding.datasets.get_dataset(args.dataset, root=os.path.expanduser('./encoding/data'),
                                           transform=transform_val, train=False, cv_val=False, hold_test=False, download=True)

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            sampler=train_sampler)

        val_sampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=args.test_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            sampler=val_sampler)

        validation_loader = {"val_loader":val_loader}
    # init the model

    model_kwargs = {}
    if args.pretrained:
        model_kwargs['pretrained'] = True

    if args.final_drop > 0.0:
        model_kwargs['final_drop'] = args.final_drop

    if args.dropblock_prob > 0.0:
        model_kwargs['dropblock_prob'] = args.dropblock_prob

    if args.last_gamma:
        model_kwargs['last_gamma'] = True

    if args.rectify:
        model_kwargs['rectified_conv'] = True
        model_kwargs['rectify_avg'] = args.rectify_avg
        
    model_kwargs['num_classes'] = args.class_num

    model = encoding.models.get_model(args.model, **model_kwargs)

    if args.dropblock_prob > 0.0:
        from functools import partial
        from encoding.nn import reset_dropblock
        nr_iters = (args.epochs - args.warmup_epochs) * len(train_loader)
        apply_drop_prob = partial(reset_dropblock, args.warmup_epochs*len(train_loader),
                                  nr_iters, 0.0, args.dropblock_prob)
        model.apply(apply_drop_prob)

    if args.mixup > 0:
        train_loader = MixUpWrapper(args.mixup, 1000, train_loader, args.gpu)
        criterion = NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        criterion = LabelSmoothing(args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    model.cuda(args.gpu)
    criterion.cuda(args.gpu)
    model = DistributedDataParallel(model, device_ids=[args.gpu])

    # criterion and optimizer
    if args.no_bn_wd:
        parameters = model.named_parameters()
        param_dict = {}
        for k, v in parameters:
            param_dict[k] = v
        bn_params = [v for n, v in param_dict.items() if ('bn' in n or 'bias' in n)]
        rest_params = [v for n, v in param_dict.items() if not ('bn' in n or 'bias' in n)]
        if args.gpu == 0:
            print(" Weight decay NOT applied to BN parameters ")
            print(f'len(parameters): {len(list(model.parameters()))} = {len(bn_params)} + {len(rest_params)}')
        optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0 },
                                     {'params': rest_params, 'weight_decay': args.weight_decay}],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    # check point
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu == 0:
                print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1 if args.start_epoch == 0 else args.start_epoch
            best_pred = checkpoint['best_pred']
            acclist_train = checkpoint['acclist_train']
            acclist_val = checkpoint['acclist_val']
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.gpu == 0:
                print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            raise RuntimeError ("=> no resume checkpoint found at '{}'".\
                format(args.resume))
    scheduler = LR_Scheduler(args.lr_scheduler,
                             base_lr=args.lr,
                             num_epochs=args.epochs,
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=args.warmup_epochs)
    def train(epoch):
        recording = open(result_file, 'a')
        target_recording = open(target_file, 'a')

        train_sampler.set_epoch(epoch)
        model.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        global best_pred, acclist_train
        for batch_idx, (data, target) in enumerate(train_loader):
            scheduler(optimizer, batch_idx, epoch, best_pred)
            if not args.mixup:
                data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            optimizer.zero_grad()
            output = model(data)
            epoch_info = 'Train | Round: %d | Epoch: %d | GPU: %d | batch_idx: %d | ' %(args.kfold, epoch, args.gpu, batch_idx)
            target_recording.write(epoch_info + str(target) + "\n")
            
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if not args.mixup:
                acc1 = accuracy(output, target, epoch_info, topk=(1,))
                top1.update(acc1[0], data.size(0))

            losses.update(loss.item(), data.size(0))
        if args.mixup:
            print('Loss: %.3f'%(losses.avg))
        else:
            print('Loss: %.3f | Top1: %.3f'%(losses.avg, top1.avg))
        string_builder = 'Train | Round: %d | Epoch: %d | GPU: %d | Loss: %.3f | Top1: %.3f'%(args.kfold, epoch, args.gpu, losses.avg, top1.avg)
        string_builder = string_builder + " | Dataset: " + args.Download_folder + " | Model: " + args.model + "\n"
        recording.write(string_builder)

        acclist_train += [float(top1.avg)]

    def validate(epoch, val_type):
        recording = open(result_file, 'a')
        target_recording = open(target_file, 'a')

        loader = validation_loader[val_type]
        model.eval()
        top1 = AverageMeter()
        global best_pred, acclist_val, cv_acclist_val
        is_best = False
        
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(args.gpu), target.cuda(args.gpu)
            
            with torch.no_grad():
                output = model(data)
                
                epoch_info = "Type: " + val_type + ' | Round: %d | Epoch: %d | GPU: %d | batch_idx: %d | ' %(args.kfold, epoch, args.gpu, batch_idx)
                target_recording.write(epoch_info + str(target) + "\n")
                
                acc1 = accuracy(output, target, epoch_info, topk=(1,))
                top1.update(acc1[0], data.size(0))

        # sum all
        sum1, cnt1 = torch_dist_sum(args.gpu, top1.sum, top1.count)

        top1_acc = sum(sum1) / sum(cnt1)

        string_builder = "Type: " + val_type + ' | Round: %d | Epoch: %d | GPU: %d | Top1: %.3f'%(args.kfold, epoch, args.gpu, top1_acc)
        string_builder = string_builder + " | Dataset: " + args.Download_folder + " | Model: " + args.model + "\n"
        recording.write(string_builder)

        if args.eval:
            if args.gpu == 0:
                print(val_type + ' Validation: Top1: %.3f'%(top1_acc))
            return

        if args.gpu == 0:
            print(val_type + ' Validation: Top1: %.3f'%(top1_acc))

            if val_type == "general_val_loader":
                acclist_val += [top1_acc]
            elif val_type == "val_loader":
                cv_acclist_val += [top1_acc]

            # save checkpoint
            if top1_acc > best_pred:
                best_pred = top1_acc
                is_best = True
            encoding.utils.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_pred,
                'acclist_train':acclist_train,
                'acclist_val':acclist_val,
                }, args=args, is_best=is_best)

    if args.export:
        if args.gpu == 0:
            torch.save(model.module.state_dict(), args.export + '.pth')
        return

    if args.eval:
        validate(args.start_epoch)
        return

    # Execution
    start = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        tic = time.time()
        train(epoch)
        acclist_train_set.append(acclist_train)
        validate(epoch, "general_val_loader")
        acclist_val_set.append(acclist_val)
        elapsed = time.time() - tic
        if args.gpu == 0:
            print(f'Epoch: {epoch}, Time cost: {elapsed}')
    validate(epoch, "val_loader")
    end = time.time()

    print(str(end - start))

    if args.gpu == 0:
        encoding.utils.save_checkpoint({
            'epoch': args.epochs-1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'acclist_train':acclist_train,
            'acclist_val':acclist_val,
            }, args=args, is_best=False)

if __name__ == "__main__":
    main()
