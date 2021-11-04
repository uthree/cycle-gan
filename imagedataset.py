import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2
from tqdm import tqdm
import os
import glob

class ImageDataset(torch.utils.data.Dataset):
    """Some Information about ImageDataset"""
    def __init__(self, dir_path,):
        print(f"loading image from {dir_path}...")
        super(ImageDataset, self).__init__()
        self.len = os.listdir(dir_path).__len__()
        self.images = [cv2.imread(path).transpose(2, 0, 1).astype(float) / 255 for path in tqdm(glob.glob(dir_path + "/*"))]
    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return self.len

ds = ImageDataset("./summer2winter_yosemite/testA/")