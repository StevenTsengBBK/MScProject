##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Modified by: Yao-I Tseng
## Email: mrsuccess1203@gmail.com
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

class ImageNetDataset(datasets.ImageFolder):
    BASE_DIR = "urbansound8k"
    def __init__(self, root=os.path.expanduser('~/encoding/data'), transform=None,
                 target_transform=None, train=True, test=False, **kwargs):
        if train:
            split = 'train'
        elif test:
            split = 'test'
        else:
            split = 'val'
        root = os.path.join(root, self.BASE_DIR, split)
        super(ImageNetDataset, self).__init__(root, transform, target_transform)
