# -*- coding: utf-8 -*-
# @Time    : 2020/11/25 13:23
# @Author  : duyu
# @Email   : abelazady@foxmail.com
# @File    : main.py
# @Software: PyCharm
import numpy as np
import torch
import torch.nn as nn
import sys
from load_data import load_data
from net import mynet
from utils import AverageMeter, LOG_INFO
import os
import pandas as pd
import csv

model_save_path = './model.pt'
config = {
    'learning_rate': 0.01,
    'batch_size': 128,
    'max_epoch': 100,
    'test_epoch': 5,
    'momentum': 0.001,
    'weight_decay':0.0001
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')

def data_split(x_data, y_data, batch_size, quota = 10,seed = 100 ):
    data_num = len(x_data)
    val_num = data_num//quota
    train_num = data_num - val_num
    np.random.seed(seed)
    i = np.random.randint(0,10)
    if i == 0:
        return x_data[i*val_num+val_num:data_num], y_data[i*val_num+val_num:data_num], x_data[i*val_num:i*val_num+val_num],\
              y_data[i*val_num:i*val_num+val_num]
    else:
        return x_data[0:i*val_num-1] + x_data[i*val_num+val_num:data_num], y_data[0:i*val_num-1] + y_data[i*val_num+val_num:data_num],\
              x_data[i*val_num:i*val_num+val_num],y_data[i*val_num:i*val_num+val_num]


def data_iterater(x_loader,y_loader,batch_size,shuffle=True):
    indx = list(range(len(x_loader)))
    if shuffle:
        np.random.shuffle(indx)
    for start in range(0, len(x_loader), batch_size):
        end = min(start + batch_size, len(x_loader))
        yield torch.Tensor(x_loader[start:end]), torch.Tensor(y_loader[start:end])

def train(x_train, y_train, model, criterion, optimizer, batch_size,epoch):
    model.train()
    for input, label in data_iterater(x_train, y_train, batch_size):
        input.resize_(input.size(0),1,input.size(1),input.size(2))
        input = input.cuda()
        label = label.cuda()
        label = label.to(dtype=torch.int64)

        y_pred = model(input)
        # y_pred = model(input).type(torch.LongTensor)
        # label = label.type(torch.LongTensor)
        loss = criterion(y_pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def validate(x_val, y_val, model, criterion):
    model.eval()
    acc_list = []
    loss_list = []
    for input, label in zip(x_val,y_val):
        input = torch.from_numpy(input)
        label = torch.Tensor([label])
        input.resize_(1,1,input.size(0),input.size(1))
        input = input.cuda()
        input = input.to(dtype=torch.float32)
        label = label.cuda()
        label = label.to(dtype=torch.int64)
        with torch.no_grad():
            output = model(input)
            loss = criterion(output,label)
            loss_list.append(loss.cpu().numpy())
            pred = torch.argmax(output)
            if pred == label:
                acc_list.append(1)
            else:
                acc_list.append(0)
    msg = 'Testing,total mean loss %.5f,total acc %.5f' %(np.mean(loss_list),np.mean(acc_list))
    LOG_INFO(msg)



def adjust_learning_rate(optimizer, epoch):
    lr = config['learning_rate'] * (0.5 ** (epoch //30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(model):
    f = open('./pred.csv','a',newline='')
    writer = csv.writer(f)
    writer.writerow(['ImageId','Label'])
    data = pd.read_csv('./digit-recognizer/test.csv')
    data_list = data.values.tolist()
    # pred_list = []
    # id_list   = []
    model.cuda()
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
        # id_list.append(id+1)
        # pred_list.append(pred.cpu().numpy())
        writer.writerow([id+1,pred.cpu().numpy()])
    f.close()
    # df = pd.DataFrame({'ImageId':id_list,'Label':pred_list})
    # df.to_csv('./pred.csv')




def main():
    x_data,y_data = load_data()

    # data_spliter = data_split(x_data,y_data,batch_size=config['batch_size'])
    model = mynet()
    model.cuda()
    # model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(),config['learning_rate'],momentum=config['momentum'],weight_decay=config['weight_decay'])
    for epoch in range(config['max_epoch']):
        print('epoch:',epoch)
        x_train, y_train, x_val, y_val = data_split(x_data,y_data,batch_size=config['batch_size'],seed=epoch)
        adjust_learning_rate(optimizer, epoch)
        train(x_train, y_train, model, criterion, optimizer, config['batch_size'] ,epoch)
        if epoch % config['test_epoch'] == 0:
            acc = validate(x_val, y_val, model, criterion)

    torch.save(model,model_save_path)
    test(model)

if __name__ == '__main__':
    main()
