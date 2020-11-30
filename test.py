# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 22:49
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : test.py
# @Software: PyCharm
from main import test
import torch

model = torch.load('./model.pt')
test(model)