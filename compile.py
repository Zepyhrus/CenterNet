import os

import numpy as np

import cv2

import torch
import torchvision.transforms as transforms
import torch.utils.data

# CenterNet modules
from src._init_paths import add_path

add_path('src/lib')

from opts import opts
from datasets.dataset.coco_hp import COCOHP
from datasets.sample.multi_pose import MultiPoseDataset
from models.networks.pose_dla_dcn import get_pose_net


# initialize options
opt = opts().parse(['multi_pose', '--batch_size', '1'])

# Creating a class inherits from both COCOHP and MultiPoseDataset
class Dataset(COCOHP, MultiPoseDataset):
  pass

opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

dataset = Dataset(opt, 'train')

train_loader = torch.utils.data.DataLoader(
  dataset, 
  batch_size=opt.batch_size, 
  shuffle=True,
  num_workers=opt.num_workers,
  pin_memory=True,
  drop_last=True
)

model = get_pose_net(34, opt.heads, opt.head_conv)


print(model)



