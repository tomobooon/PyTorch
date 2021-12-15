#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import cv2
import random
from multiboxloss import MultiBoxLoss

## (1) �f�[�^�̏����Ɛݒ�
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 30
epoch_num = 15
ans = pickle.load(open('ans.pkl', 'rb'))
files = list(ans.keys())
datanum = len(files)
dirpath = './VOCdevkit/VOC2012/JPEGImages/'

## (2) ���f���̒�`
from mynet import SSD

## (3) ���f���̐����C�����֐��C�œK���֐��̐ݒ�
##   (3.1) ���f���̐���
net = SSD()
net.to(device)

##   (3.2) �����֐� �̐ݒ�
optimizer = optim.SGD(net.parameters(),
                      lr=1e-3,momentum=0.9,
                      weight_decay=5e-4)

##   (3.3) �œK���֐��̐ݒ�
from multiboxloss import MultiBoxLoss
criterion = MultiBoxLoss(device=device)

## (4) �w�K
net.train()
for ep in range(epoch_num):
    random.shuffle(files)
    xs, ys, bc = [], [], 0
    for i in range(datanum):
        file = files[i]
        if (bc < batch_size):
            filename = str(dirpath) + str(file) + ".jpg"
            image = cv2.imread(filename)
            x = cv2.resize(image, (300, 300)).astype(np.float32)
            x -= (104.0, 117.0, 123.0)
            x = torch.from_numpy(x[:,:,(2,1,0)]).permute(2,0,1)
            y =  ans[file]
            xs.append(torch.FloatTensor(x))
            ys.append(torch.FloatTensor(y))
            bc += 1
        if ((bc == batch_size) or (i == datanum - 1)):
            images = torch.stack(xs, dim=0)
            images = images.to(device)
            targets = [ y.to(device) for y in ys ]
            outputs = net(images)
            loss_l, loss_c = criterion(outputs, targets)
            loss = loss_l + loss_c
            print(i, loss_l.item(), loss_c.item())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
            optimizer.step()
            loss_l, loss_c  = 0, 0
            xs, ys, bc = [], [], 0

## (5) ���f���̕ۑ��i�e epoch �Ń��f����ۑ��j
    outfile = "ssd0-" + str(ep) + ".model"
    torch.save(net.state_dict(),outfile)
    print(outfile," saved")
