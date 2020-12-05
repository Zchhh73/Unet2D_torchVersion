from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from skimage.io import imread
from skimage import color
from PIL import Image
from skimage import color
import os
import random


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


class VerseDataset(Dataset):
    def __init__(self, args, img_path, mask_path, aug=False):
        self.args = args
        self.img_paths = img_path
        self.mask_paths = mask_path
        self.aug = aug
        imgs = []
        for i in range(len(img_path)):
            img = img_path[i]
            mask = mask_path[i]
            imgs.append([img, mask])

        self.imgs = imgs

    # 数据集长度
    def __len__(self):
        return len(self.imgs)

    # 数据集item
    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = imread(x_path)
        img_y = imread(y_path)

        img_x = img_x.astype('float32') / 255
        img_y = img_y.astype('float32') / 255

        if self.aug:
            if random.uniform(0, 1) > 0.5:
                # 图片向右翻转180°
                img_x = img_x[:, ::-1, :].copy()
                img_y = img_y[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                # 图片向下翻转180°
                img_x = img_x[::-1, :, :].copy()
                img_y = img_y[::-1, :].copy()
        # img_x = color.gray2rgb(img_x)
        # image = image[:,:,np.newaxis]
        img_x = img_x.transpose((2, 0, 1))
        # img_y = img_y[:, :, np.newaxis]
        img_y = img_y.transpose((2, 0, 1))
        return img_x, img_y
