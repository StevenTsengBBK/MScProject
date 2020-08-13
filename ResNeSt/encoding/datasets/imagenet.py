##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

class ImageNetDataset(datasets.ImageFolder):
    BASE_DIR = "urbansound8k"
    def __init__(self, root=os.path.expanduser('./encoding/data'), transform=None,
                 target_transform=None, train=True, cv_val = False, hold_test = False,  **kwargs):
        if train:
            split = 'train'
        elif hold_test:
            split = 'test'
        elif cv_val:
            split = 'CV'
        else:
            split = 'val'
            
        root = os.path.join(root, self.BASE_DIR, split)
        super(ImageNetDataset, self).__init__(
            root, transform, target_transform)
