# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 22:49
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : test.py
# @Software: PyCharm
import torch
#from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import net
import csv
import os
summary_writer = SummaryWriter('runs/model_structure')
output_path = 'best_pred.csv'
# model = torch.load('./mynet1/mynet1_epoch100.pt')
model = net.__dict__['Rnn']()
model.cpu()
x = torch.randn(1, 1, 28, 28).cpu()
summary_writer.add_graph(model, x)
summary_writer.close()
# net_plot = make_dot(model(x), params=dict(model.named_parameters()))
# net_plot.view()

# tw.draw_model(model, [1, 1, 28, 28])
#if hasattr(torch.cuda, 'empty_cache'):
#	torch.cuda.empty_cache()
