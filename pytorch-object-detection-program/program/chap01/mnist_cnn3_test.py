#!/usr/bin/python
# -*- coding: sjis -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import sys

argvs = sys.argv
argc = len(argvs)

# GPU/CPU device setting

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data setting

dataset = datasets.MNIST('./data', train=False, download=True)
xt = dataset.data.reshape(10000,1,28,28) / 255.0
yans1 = dataset.targets
yans2 = dataset.targets
for i in range(len(yans2)):
    if (yans2[i] in [6, 8, 9]):
        yans2[i] = 1
    else:
        yans2[i] = 0

# Define model

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.cn1 = nn.Conv2d(1, 20, 5)  
        self.pool1 = nn.MaxPool2d(2)    
        self.cn2 = nn.Conv2d(20, 50, 5) 
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(3200, 10)  
    def forward(self, x):
        x = F.relu(self.cn1(x))
        x = self.pool1(x)
        x = F.relu(self.cn2(x))
        x = self.dropout(x)   
        x = x.view(len(x), -1) 
        return self.fc(x)

class MyCNN2(nn.Module):
    def __init__(self):
        super(MyCNN2, self).__init__()
        self.fc = nn.Linear(3200, 2)  
    def forward(self, x):
        return self.fc(x)

# Initialize model

model1 = MyCNN()
model2 = MyCNN2()

# Load model

model1.load_state_dict(torch.load(argvs[1]))
model2.load_state_dict(torch.load(argvs[2]))

# Test

def fwd(model, x):
    x = F.relu(model.cn1(x))
    x = model.pool1(x)
    x = F.relu(model.cn2(x))
    x = model.dropout(x)   
    x = x.view(len(x), -1) 
    return x

model1.eval()
model2.eval() 
with torch.no_grad():
    cnnx = fwd(model1, xt) 
    out1 = model1.fc(cnnx)
    out2 = model2(cnnx)    
    ans1 = torch.argmax(out1,1)
    ans2 = torch.argmax(out2,1)    
    print(((yans1 == ans1).sum().float()/len(ans1)).item())
    print(((yans2 == ans2).sum().float()/len(ans2)).item())    





