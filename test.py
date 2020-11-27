from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import os

img_path = "D:\\train_data\\img"
mask_path = "D:\\train_data\\mask"


class DataSet_Verse(Dataset):
    def __init__(self, img_path, mask_path, transform=None, target_transform=None):
        imgs = []
        for filename in os.listdir(img_path):
            img = os.path.join(img_path, filename)
            mask = os.path.join(mask_path, filename)
            imgs.append([img, mask])

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform


if __name__ == '__main__':
    imgs = []
    for filename in os.listdir(img_path):
        img = os.path.join(img_path, filename)
        mask = os.path.join(mask_path, filename)
        imgs.append([img,mask])

    print(imgs)
