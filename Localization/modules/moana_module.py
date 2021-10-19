import torch
import torch.nn as nn
import cv2
import numpy as np

def conv1x3(in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
    """1x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size - 1) // 2, groups=groups, bias=False, dilation=dilation)

class moana(nn.Module):
    def __init__(self, channel, k_size=3):
        super(moana, self).__init__()

        self.avg_pool_c = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.avg_pool_h = nn.AdaptiveAvgPool3d((None, None, 1))
        self.avg_pool_w = nn.AdaptiveAvgPool3d((None, 1, None))

        self.conv_c = conv1x3(1, 1, kernel_size=k_size)
        self.conv_h = conv1x3(channel, channel, kernel_size=k_size)
        self.conv_w = conv1x3(channel, channel, kernel_size=k_size)

        self.sigmoid = nn.Sigmoid()

        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.softmax_c = nn.Softmax(dim=1)
        self.softmax_h = nn.Softmax(dim=-2)
        self.softmax_w = nn.Softmax(dim=-1)

    def forward(self, x):
        y_c = self.avg_pool_c(x)
        y_h = self.avg_pool_h(x)
        y_w = self.avg_pool_w(x)

        y_c = self.bn1(self.conv_c(y_c.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))
        y_h = self.bn2(self.conv_h(y_h.squeeze(-1)).unsqueeze(-1))
        y_w = self.bn3(self.conv_w(y_w.squeeze(-2)).unsqueeze(-2))

        y_c = self.sigmoid(y_c)
        y_h = self.sigmoid(y_h)
        y_w = self.sigmoid(y_w)
        attmap = y_w + y_h + y_c


        return x * (1 + attmap)
