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

#best 0.99335 fd batchsize 64 #0.99314 bs128
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
                #m.weight.data.uniform_(0, 0.001)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        # self.featuremap = x.detach().cpu()
        x = self.classifier(x)
        return x
#0.99292 bs64 #5
class mynet2(nn.Module):
    def __init__(self):
        super(mynet2, self).__init__()
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

class mynet3(nn.Module):
    def __init__(self):
        super(mynet3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
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

class Rnn(nn.Module):
        def __init__(self, in_dim=28, hidden_dim=128, n_layer=2, n_classes=10):
            super(Rnn, self).__init__()
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
        super(mlp, self).__init__()
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

class mlp1(nn.Module):
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

class mlp2(nn.Module):
    def __init__(self):
        super(mlp2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=784, out_features=512),
            nn.BatchNorm1d(512),  # 在输出通道上做归一化
            nn.LayerNorm(512),
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


# input data shape(N=64,C=1,H=28,W=28)
class VGG_simple(nn.Module):
    def __init__(self):
        super(VGG_simple, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d((2, 2)),
            # Flatten(),

        )
        self.layer2 = nn.Sequential(
            nn.Linear(128 * 7 * 7, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(400, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        x = self.layer2(x)
        return x


'''GoogLeNet with PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


# 编写卷积+bn+relu模块
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channals, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channals, **kwargs)
        self.bn = nn.BatchNorm2d(out_channals)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


# 编写Inception模块
class Inception(nn.Module):
    def __init__(self, in_planes,
                 n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = BasicConv2d(in_planes, n1x1, kernel_size=1)

        # 1x1 conv -> 3x3 conv branch
        self.b2_1x1_a = BasicConv2d(in_planes, n3x3red,
                                    kernel_size=1)
        self.b2_3x3_b = BasicConv2d(n3x3red, n3x3,
                                    kernel_size=3, padding=1)

        # 1x1 conv -> 3x3 conv -> 3x3 conv branch
        self.b3_1x1_a = BasicConv2d(in_planes, n5x5red,
                                    kernel_size=1)
        self.b3_3x3_b = BasicConv2d(n5x5red, n5x5,
                                    kernel_size=3, padding=1)
        self.b3_3x3_c = BasicConv2d(n5x5, n5x5,
                                    kernel_size=3, padding=1)

        # 3x3 pool -> 1x1 conv branch
        self.b4_pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.b4_1x1 = BasicConv2d(in_planes, pool_planes,
                                  kernel_size=1)

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2_3x3_b(self.b2_1x1_a(x))
        y3 = self.b3_3x3_c(self.b3_3x3_b(self.b3_1x1_a(x)))
        y4 = self.b4_1x1(self.b4_pool(x))
        # y的维度为[batch_size, out_channels, C_out,L_out]
        # 合并不同卷积下的特征图
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = BasicConv2d(1, 192,
                                      kernel_size=3, padding=1)

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.linear(out)
        return out






