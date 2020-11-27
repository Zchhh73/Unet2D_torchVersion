from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import os


class DatasetVerse(Dataset):
    def __init__(self, img_path, mask_path, transform=None, target_transform=None):
        imgs = []
        for filename in os.listdir(img_path):
            img = os.path.join(img_path, filename)
            mask = os.path.join(mask_path, filename)
            imgs.append([img, mask])

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    # 数据集长度
    def __len__(self):
        return len(self.imgs)

    # 数据集item
    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y
