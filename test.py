# -*- coding: utf-8 -*-
# @Time    : 2020/11/30 22:49
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : test.py
# @Software: PyCharm
from main import test
import torch
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
import net
# import tensorwatch as tw
summary_writer = SummaryWriter('runs/model_structure')

# model = torch.load('./mynet1/mynet1_epoch100.pt')
checkpoint = torch.load('./model_epoch100.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])

model = net.__dict__['mynet1']()
model.cpu()
x = torch.randn(1, 1, 28, 28).cpu()
summary_writer.add_graph(model, x)
summary_writer.close()
# net_plot = make_dot(model(x), params=dict(model.named_parameters()))
# net_plot.view()

# test(model)
# tw.draw_model(model, [1, 1, 28, 28])
if hasattr(torch.cuda, 'empty_cache'):
	torch.cuda.empty_cache()
