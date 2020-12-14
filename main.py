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
import net
from utils import AverageMeter, LOG_INFO
import os
import pandas as pd
import csv
from logger import Logger
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description='PyTorch Digit Recoginer Training')
parser.add_argument('-e','--epoch',default=100,type=int,metavar='N',help='max epoch')
parser.add_argument('-w','--wd',default=0.001,type=float,metavar='beta',help='weight decay')
# from torchviz import make_dot
#logging.basicConfig(level=logging.DEBUG, filename='new.log',format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
path = os.path.abspath(os.path.dirname(__file__))
sys.stdout = Logger(path+'/log.txt')
summary_writer = SummaryWriter('runs/dy')
losses = AverageMeter()

net_list = {0:'mynet',1:'mynet1',2:'mynet2',3:'Rnn',4:'mlp',5:'VGG_simple'}
# net_option = net_list[1]
net_option = 'mynet1'
model_save_path = './model.pt'
output_path = net_option+'pred.csv'
best_acc = 0

if os.path.exists(output_path):
    os.remove(output_path)

# if os.path.exists('runs/dy'):
#     os.system('rm -rf runs/dy')

config_formynet1 = {
    'learning_rate': 0.01,
    'batch_size': 64,
    'max_epoch': 100,
    'test_epoch': 5,
    'momentum': 0.002,
    'weight_decay':0.0001,
}
config_formynet2 = {
    'learning_rate': 0.01,
    'batch_size': 128,
    'max_epoch': 100,
    'test_epoch': 5,
    'momentum': 0.001,
    'weight_decay':0.0001
}
config_formlp = {
    'learning_rate': 0.01,
    'batch_size': 128,
    'max_epoch': 100,
    'test_epoch': 5,
    'momentum': 0.001,
    'weight_decay':0.0001
}
config_forlstm = {
    'learning_rate': 0.01,
    'batch_size': 128,
    'max_epoch': 500,
    'test_epoch': 5,
    'momentum': 0.001,
    'weight_decay':0.0001
}
args = parser.parse_args()
config = config_formynet1
#config['max_epoch'] = args.epoch
#config['weight_decay'] = args.wd
print(net_option)
print(config)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_split(x_data, y_data, batch_size, quota = 10,seed = 100 ):
    data_num = len(x_data)
    val_num = data_num//quota
    train_num = data_num - val_num
    np.random.seed(seed)
    i = np.random.randint(0,quota)
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
        # print(loss)
        losses.update(loss.item(),input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    summary_writer.add_scalar('loss',losses.avg,epoch)



def validate(x_val, y_val, model, criterion):
    global best_acc
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
            acc_list.append(pred.eq(label).sum())
            #if pred == label:
            #    acc_list.append(1)
            #else:
            #    acc_list.append(0)
    acc = torch.mean(torch.Tensor(acc_list))
    msg = 'Testing,total mean loss %.5f,total acc %.5f,best acc %.5f' %(np.mean(loss_list),acc,best_acc)
    LOG_INFO(msg)
    return acc



def adjust_learning_rate(optimizer, epoch):
    lr = config['learning_rate'] * (0.5 ** (epoch //30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(model):
    f = open(output_path,'a',newline='')
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

def load_checkpoint(model,optimizer,path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch

def save_checkpoint(model, optimizer, epoch, path_prefix='./'):
    if not os.path.exists(path_prefix):
        os.system('mkdir -p '+ path_prefix)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, os.path.join(path_prefix, net_option + '_' + 'epoch' + str(epoch) + '.pt'))
def save_best(model, optimizer, epoch, path_prefix='./'):
    if not os.path.exists(path_prefix):
        os.system('mkdir -p '+ path_prefix)
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, os.path.join(path_prefix, net_option + '_best' + '.pt'))


def main():
    x_data,y_data = load_data()

    # data_spliter = data_split(x_data,y_data,batch_size=config['batch_size'])
    model = net.__dict__[net_option]()
    model.cuda()
    best_epoch = 0
    # model.to(device)
    global best_acc
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(),config['learning_rate'],momentum=config['momentum'],weight_decay=config['weight_decay'])
    # optimizer = torch.optim.Adam(model.parameters(),config['learning_rate'],momentum=config['momentum'],weight_decay=config['weight_decay'])
    start_epoch = 0
    #start_epoch = load_checkpoint(model,optimizer,'./mynet1/mynet1_epoch50.pt')
    x_train, y_train, x_val, y_val = data_split(x_data, y_data, batch_size=config['batch_size'], quota=10,seed=10)
    for epoch in tqdm(range(start_epoch,config['max_epoch'])):
        print('epoch:',epoch)
        adjust_learning_rate(optimizer, epoch)
        train(x_train, y_train, model, criterion, optimizer, config['batch_size'] ,epoch)
        #train(x_data, y_data, model, criterion, optimizer, config['batch_size'] ,epoch)
        if (epoch+1) % config['test_epoch'] == 0:
            acc = validate(x_val, y_val, model, criterion)
            if best_acc<acc:
                best_acc = acc
                best_epoch = epoch
                save_best(model,optimizer,epoch,'./best')
        if epoch % 50 == 0:
            if not os.path.exists('./'+net_option):
                os.system('mkdir '+net_option)
            save_checkpoint(model,optimizer,epoch,'./'+ net_option)
        if epoch == 90:
            if not os.path.exists('./'+net_option):
                os.system('mkdir '+net_option)
            save_checkpoint(model,optimizer,epoch,'./'+ net_option)
    save_checkpoint(model,optimizer,epoch,'./'+ net_option)
    acc = validate(x_val, y_val, model, criterion)
    if best_acc<acc:
        best_acc = acc
        best_epoch = epoch
        save_best(model,optimizer,epoch,'./best')

    print(config)
    print(net_option)
    print('best acc: ',best_acc)

    torch.save(model,model_save_path)

    # net_plot = make_dot(model(x),params = dict(model.named_parameters()))
    test(model)
    model.cpu()
    x = torch.randn(1, 1, 28, 28)
    summary_writer.add_graph(model, (x,))

if __name__ == '__main__':
    main()
