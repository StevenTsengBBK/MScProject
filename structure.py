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

# model including resnest50 resnest101 resnest200 resnest269 resnest50_fast resnest101_fast

# model = encoding.models.get_model("resnest50")
# model = encoding.models.get_model("resnest101")
# model = encoding.models.get_model("resnest200")
model = encoding.models.get_model("resnest269")

print(model)
