import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from model.DAUnet.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import math
import cv2

from model.DAUnet.attentionBlocks import DualAttBlock
from utils.utils import str2bool, count_params

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        if is_deconv:
            self.block = nn.Sequential(
                conv3x3_bn_relu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                conv3x3_bn_relu(in_channels, middle_channels),
                conv3x3_bn_relu(middle_channels, out_channels),
            )

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.block(x)


class DAUNet(nn.Module):
    def __init__(self, num_classes=3, num_filters=32, pretrained=True, is_deconv=True):
        super(DAUNet, self).__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.densenet121(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Encoder
        self.conv1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0)
        self.conv2 = self.encoder.features.denseblock1
        self.conv2t = self.encoder.features.transition1
        self.conv3 = self.encoder.features.denseblock2
        self.conv3t = self.encoder.features.transition2
        self.conv4 = self.encoder.features.denseblock3
        self.conv4t = self.encoder.features.transition3
        self.conv5 = nn.Sequential(self.encoder.features.denseblock4,
                                   self.encoder.features.norm5)

        # Decoder
        self.center = conv3x3_bn_relu(1024, num_filters * 8 * 2)
        self.dec5 = DualAttBlock(inchannels=[512, 1024], outchannels=512)
        self.dec4 = DualAttBlock(inchannels=[512, 512], outchannels=256)
        self.dec3 = DualAttBlock(inchannels=[256, 256], outchannels=128)
        self.dec2 = DualAttBlock(inchannels=[128, 128], outchannels=64)
        self.dec1 = DecoderBlock(64, 48, num_filters, is_deconv)
        self.dec0 = conv3x3_bn_relu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)

    def forward(self, x):
        x_size = x.size()
        # encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2t(self.conv2(conv1))
        conv3 = self.conv3t(self.conv3(conv2))
        conv4 = self.conv4t(self.conv4(conv3))
        conv5 = self.conv5(conv4)
        # decoder
        conv2 = F.interpolate(conv2, scale_factor=2, mode='bilinear', align_corners=True)
        conv3 = F.interpolate(conv3, scale_factor=2, mode='bilinear', align_corners=True)
        conv4 = F.interpolate(conv4, scale_factor=2, mode='bilinear', align_corners=True)
        center = self.center(self.pool(conv5))
        dec5, _ = self.dec5([center, conv5])
        dec4, _ = self.dec4([dec5, conv4])
        dec3, att = self.dec3([dec4, conv3])
        dec2, _ = self.dec2([dec3, conv2])
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        out = self.final(dec0)
        att = F.interpolate(att, scale_factor=4, mode='bilinear', align_corners=True)
        return out

    def pad(self, x, y):
        diffX = y.shape[3] - x.shape[3]
        diffY = y.shape[2] - x.shape[2]

        return nn.functional.pad(x, (diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2))


if __name__ == '__main__':
    model = DAUNet().cuda()
    print("参数：%.2f" % (count_params(model) / (1024 * 1024)) + "MB")
    data = torch.randn((1, 3, 128, 128)).cuda()
    pred = model(data)
    print(pred.shape)