# https://pytorch.apachecn.org/docs/1.0/blitz_autograd_tutorial.html
from __future__ import print_function
import torch

# x = torch.randn(4, 4)
#
# print(x)

# 张量

# 数据加载和处理教程
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode



