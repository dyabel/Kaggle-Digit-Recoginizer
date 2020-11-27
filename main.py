# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 13:23
# @Author  : Mat
# @Email   : mat_wu@163.com
# @File    : main.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn
import torch.nn as nn
from load_data import load_data


def data_iterator(x, y, batch_size, shuffle=True):
    indx = list(range(len(x)))
    if shuffle:
        np.random.shuffle(indx)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        yield x[indx[start:end]], y[indx[start:end]]


def main():
    load_data()
    pass


if __name__ == '__main__':
    main()
