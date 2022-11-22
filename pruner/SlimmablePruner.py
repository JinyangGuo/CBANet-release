import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SubsetRandomSampler, SequentialSampler
from torch.autograd import Variable
from torchvision import models

import os
import sys
import numpy as np
import socket
import argparse
import time
import random
import math
import copy
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm
from operator import itemgetter
from heapq import nsmallest
from thop import profile

sys.path.append('..')
sys.path.append('.')
import model_slimmable
import models.analysis as analysis
import models.synthesis as synthesis
import models.analysis_prior as analysis_prior
import models.synthesis_prior as synthesis_prior
import models.synthesis_slimmable as synthesis_slimmable
from models.GDN import GDN
from models.ms_ssim_torch import ms_ssim


class Gate(nn.Module):
    def __init__(self):
        super(Gate, self).__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, input):
        return self.weight * input


class AQL(nn.Module):
    def __init__(self, in_channel):
        super(AQL, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 192, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2, groups=192)
        self.conv3 = nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2, groups=192)
        self.conv4 = nn.Conv2d(192, in_channel, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.sigmoid(self.conv4(x))
        x = identity - identity * x
        return x


class IAQL(nn.Module):
    def __init__(self, in_channel):
        super(IAQL, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 192, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2, groups=192)
        self.conv3 = nn.Conv2d(192, 192, kernel_size=5, stride=1, padding=2, groups=192)
        self.conv4 = nn.Conv2d(192, in_channel, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.relu(self.conv4(x))
        x = identity + identity * x
        return x

