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
# import tensorwatch as tw
summary_writer = SummaryWriter('runs/model_structure')
output_path = 'best_pred.csv'
if os.path.exists(output_path):
    os.system('rm '+output_path)
    print('remove old pred')
config = {
    'learning_rate': 0.01,
    'batch_size': 64,
    'max_epoch': 100,
    'test_epoch': 5,
    'momentum': 0.001,
    'weight_decay':0.0001,
}
# model = torch.load('./mynet1/mynet1_epoch100.pt')
model = net.__dict__['mynet1']()
optimizer = torch.optim.SGD(model.parameters(),config['learning_rate'],momentum=config['momentum'],weight_decay=config['weight_decay'])
checkpoint = torch.load('./best/mynet1_best.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
model.cuda()
#x = torch.randn(1, 1, 28, 28).cpu()
#summary_writer.add_graph(model, x)
#summary_writer.close()
# net_plot = make_dot(model(x), params=dict(model.named_parameters()))
# net_plot.view()

# tw.draw_model(model, [1, 1, 28, 28])
#if hasattr(torch.cuda, 'empty_cache'):
#	torch.cuda.empty_cache()
f = open(output_path,'a',newline='')
writer = csv.writer(f)
writer.writerow(['ImageId','Label'])
data = pd.read_csv('./digit-recognizer/test.csv')
data_list = data.values.tolist()
model.eval()
for id,line in enumerate(data_list):
    image_1d = np.array(line)
    image_2d = image_1d.reshape(28, 28)
    input = torch.Tensor(image_2d)
    input.resize_(1,1,input.size(0),input.size(1))
    input = input.cuda()
    input = input.to(dtype=torch.float32)
    output = model(input)
    pred = torch.argmax(output)
    writer.writerow([id+1,pred.cpu().numpy()])
f.close()
