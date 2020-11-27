# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 13:39
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : load_data.py
# @Software: PyCharm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms,utils

def load_data():
    data = pd.read_csv('./digit-recognizer/train.csv')
    data_list = data.values.tolist()
    train_loader = []
    # cnt = 0
    print(len(data_list))
    for line in data_list:
        # cnt += 1
        # if cnt == 100:
        #     break
        label = line[0]
        image_1d = torch.Tensor(line[1::])
        image_2d = image_1d.reshape(28,28)
        # grid = utils.make_grid(image_2d)
        # grid = grid.numpy().transpose(1,2,0)
        # std = [0.5]
        # mean = [0.5]
        # grid = grid * std + mean
        train_loader.append((image_1d,label))
        # '''
        # fig1 = plt.figure(1)
        # print(grid.shape)
        # plt.imshow(grid)
        # plt.draw()
        # print(label)
        # plt.pause(1)
        # plt.close(fig1)
        # '''
    return train_loader

