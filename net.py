# -*- coding: utf-8 -*-
# @Time    : 2020/11/27 15:49
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : net.py
# @Software: PyCharm
import torch.nn as nn
import torch.nn.init as init
import math

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1568, 1000),
            nn.ReLU(True),
            nn.Linear(1000,100),
            nn.ReLU(True),
            nn.Linear(100, 10),
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        # self.featuremap = x.detach().cpu()
        x = self.classifier(x)
        return x

class mynet1(nn.Module):
    def __init__(self):
        super(mynet1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1568, 1000),
            nn.ReLU(True),
            nn.Linear(1000,100),
            nn.ReLU(True),
            nn.Linear(100, 10),
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        # self.featuremap = x.detach().cpu()
        x = self.classifier(x)
        return x
