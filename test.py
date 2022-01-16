from genericpath import getsize
import os
import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import unittest
import torch
import torch.nn as nn
import torchvision
import json


from dataset import ImagenetteDataset as datasetclass
from dataset import get_statistics, Augmentations

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()

print(device)

dataset = datasetclass(r"C:\Users\Admin\Documents\Technion\Tutorials\Advance Deep Learning\hw\imagenette2", normalize=False, augment=1, num_augmentations = 2, crop_size=114)

img = dataset[1500]


print(img[1].shape) #should be (3,244,244)

#print(get_statistics(dataset))

f, axarr = plt.subplots(2,1)
axarr[0].imshow(img[0].permute(1,2,0).numpy())
axarr[1].imshow(img[1].permute(1,2,0).numpy())
plt.show()