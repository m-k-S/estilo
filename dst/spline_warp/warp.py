import os
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch.autograd import Variable
from PIL import Image
from imageio import imread, imwrite

import utils_misc
from utils_warp import _image_warp, apply_warp


### Define paths
content_path = 'original_image.png'
src_pts_path = 'src_Kpts.txt'
target_pts_path = 'target_Kpts.txt'
output_name = 'warped_image.png'


### Read in the image and the source, destination/target keypoints
content_im = utils_misc.to_device(Variable(utils_misc.load_path_for_pytorch(content_path, 256).unsqueeze(0)))
_, _, height, width = content_im.size()

src_pts = torch.from_numpy(np.asarray(np.loadtxt(src_pts_path, delimiter=','))).float()
target_pts = torch.from_numpy(np.asarray(np.loadtxt(target_pts_path, delimiter=','))).float()


### Add points at the image boundary to prevent weird warps
num_pts = 80
w_d = width//(num_pts+1)
w_pts = w_d*(np.arange(num_pts)+1)
h_d = height//(num_pts+1)
h_pts = h_d*(np.arange(num_pts)+1)

border_pts = [[0, 0], [height-1, 0], [0, width-1], [height-1, width-1]]
for i in range(10):
    border_pts.append([h_pts[i], 0])
    border_pts.append([h_pts[i], width-1])
    border_pts.append([0, w_pts[i]])
    border_pts.append([height-1, w_pts[i]])
border_pts = torch.from_numpy(np.asarray(border_pts)).float()

no_flow = [[0., 0.]] * len(border_pts)
no_flow = torch.from_numpy(np.asarray(no_flow)).float()

src_pts_aug = torch.cat([src_pts, border_pts], 0).cuda()
src_pts_aug = src_pts_aug/torch.max(src_pts_aug,0,keepdim=True)[0]

dst_pts_aug = torch.cat([target_pts, border_pts], 0).cuda()
dst_pts_aug = dst_pts_aug/torch.max(dst_pts_aug,0,keepdim=True)[0]


### Warp the content image such that source points -> destination points
im_warp = apply_warp(content_im, [src_pts_aug], [dst_pts_aug])


### Output the warped image
canvas = torch.clamp(im_warp[0][0], 0., 1.).data.cpu().numpy().transpose(1,2,0)
imwrite(output_name, canvas)
