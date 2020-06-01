import time
import shutil
import os
import sys

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from imageio import imread

use_gpu = True

def match_device(ref, mut):
    if ref.is_cuda and not mut.is_cuda:
        mut = mut.cuda()

    if not ref.is_cuda and mut.is_cuda:
        mut = mut.cpu()

    return mut

def to_device(tensor):
    if use_gpu:
        return tensor.cuda()
    else:
        return tensor

def match_device(ref, mut):
    if ref.is_cuda and not mut.is_cuda:
        mut = mut.cuda()

    if not ref.is_cuda and mut.is_cuda:
        mut = mut.cpu()

    return mut

def load_path_for_pytorch(path, max_side=1000, verbose=False):

    com_f = max

    x = imread(path)
    s = x.shape

    x = x/255.
    xt = x.copy()

    if len(s) < 3:
        x = np.stack([x,x,x],2)

    if x.shape[2] > 3:
        x = x[:,:,:3]

    x = x.astype(np.float32)
    x = torch.from_numpy(x).contiguous().permute(2,0,1).contiguous()

    if (com_f(s[:2])>max_side and max_side>0):
        fac = float(max_side)/com_f(s[:2])
        x = F.interpolate(x.unsqueeze(0),( int(s[0]*fac), int(s[1]*fac) ), mode='bilinear', align_corners=True)[0]
        so = s

    s = x.shape
    if verbose:
        print('DEBUG: image from path {} loaded with size {}'.format(path,s))

    return x
