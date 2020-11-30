# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 21:51
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : utils.py
# @Software: PyCharm
from datetime import datetime


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print(display_now + ' ' + msg)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
