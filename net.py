# -*- coding: utf-8 -*-
# @Time    : 2020/11/27 15:49
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : net.py
# @Software: PyCharm
import torch.nn as nn
import torch.nn.init as init
import math
import torch

#best 0.99335 fd batchsize 64 #0.99314 bs128
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
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
                #m.weight.data.uniform_(0, 0.001)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        # self.featuremap = x.detach().cpu()
        x = self.classifier(x)
        return x
#0.99292 bs64 #5
class cnn_with_bn(nn.Module):
    def __init__(self):
        super(cnn_with_bn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1568, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True),
            nn.Linear(1000,100),
            nn.BatchNorm1d(100),
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

class rnn(nn.Module):
        def __init__(self, in_dim=28, hidden_dim=128, n_layer=2, n_classes=10):
            super(rnn, self).__init__()
            self.n_layer = n_layer
            self.hidden_dim = hidden_dim
            self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
            self.classifier = nn.Linear(hidden_dim, n_classes)

        def forward(self, x):
            x = x.view(x.size(0),x.size(2),x.size(3))
            out, (h_n, c_n) = self.lstm(x)
            # 此时可以从out中获得最终输出的状态h
            # x = out[:, -1, :]
            x = h_n[-1, :, :]
            x = self.classifier(x)
            return x

class mlp(nn.Module):
    def __init__(self):
        super(mlp1, self).__init__()
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(784, 500),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Linear(500,100),
            nn.ReLU(True),
            nn.Linear(100, 10),
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

class mlp_with_bn(nn.Module):
    def __init__(self):
        super(mlp_with_bn, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.BatchNorm1d(512),  # 在输出通道上做归一化
            #nn.LayerNorm(512),
            # nn.InstanceNorm1d(512),
            #nn.GroupNorm(2, 512),

            nn.ReLU(inplace=False)  # inplace是否释放内存
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x):  # 实例化网络时，call方法直接调用forward。
        x = torch.reshape(x, [x.size(0), -1])  # 将数据形状转成N,V结构 , V=C*H*W
        y1 = self.fc1(x)
        y2 = self.fc2(y1)
        y3 = self.fc3(y2)
        self.y4 = self.fc4(y3)
        output = torch.softmax(self.y4, 1)  # 输出（N, 10）.第0轴是批次，第一轴是数据. 作用：将实数值转为概率值
        return output



