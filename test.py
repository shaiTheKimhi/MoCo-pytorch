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
from dataset import get_statistics 


seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams.update({'font.size': 12})
test = unittest.TestCase()

print(device)

dataset = datasetclass(r"C:\Users\Admin\Documents\Technion\Tutorials\Advance Deep Learning\hw\imagenette2")

img = dataset[4][0]

print(img.shape) #should be (3,244,244)

print(get_statistics(dataset))

#plt.imsave(r"C:\Users\Admin\Documents\Technion\Tutorials\Advance Deep Learning\hw\hw1\temp.png", img.permute(1,2,0).numpy())