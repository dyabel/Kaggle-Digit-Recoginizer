# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 13:39
# @Author  : duyu
# @Email   : abelazady@foxmail.com
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
    # cnt = 0
    x_loader = []
    y_loader = []
    print('data length:', len(data_list))
    for line in data_list:
        # cnt += 1
        # if cnt == 100:
        #     break
        label = line[0]
        image_1d = np.array(line[1::])
        image_2d = image_1d.reshape(28,28)
        x_loader.append(image_2d)
        y_loader.append(label)
        # grid = utils.make_grid(image_2d)
        # grid = grid.numpy().transpose(1,2,0)
        # std = [0.5]
        # mean = [0.5]
        # grid = grid * std + mean
        # '''
        # fig1 = plt.figure(1)
        # print(grid.shape)
        # plt.imshow(image_2d)
        # plt.draw()
        # print(label)
        # plt.pause(1)
        # plt.close(fig1)
        # '''
    return x_loader,y_loader

