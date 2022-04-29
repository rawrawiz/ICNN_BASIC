# -*- coding: utf-8 -*-

import h5py
import math
import copy
import scipy.io as io
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.alexnet.conv_mask import conv_mask



class alexnet(nn.Module):
    def __init__(self, pretrain_path, label_num, dropoutrate, losstype):
        super(alexnet, self).__init__()
        self.pretrian_path = pretrain_path
        self.dropoutrate = dropoutrate
        self.label_num = label_num
        self.losstype = losstype
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.00002, beta=0.75, k=1.0),)
        self.maxpool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False), )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.00002, beta=0.75, k=1.0),)
        self.maxpool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False), )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), )
        
        self.maxpool3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False), )
        
   
        self.relu = nn.Sequential(
            nn.ReLU(inplace=True), )
        
        self.line = nn.Sequential(
            nn.Dropout2d(p=self.dropoutrate),
            nn.Conv2d(256,256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Flatten(start_dim=1, end_dim=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.dropoutrate),
            nn.Conv2d(4096, self.label_num, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)), )
        self.init_weight()

    def init_weight(self):
        data = loadmat(self.pretrian_path)
        w, b = data['layers'][0][0][0]['weights'][0][0]
        self.conv1[0].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv1[0].bias.data.copy_(torch.from_numpy(b.reshape(-1)))

        w, b = data['layers'][0][4][0]['weights'][0][0]
        w = w.transpose([3, 2, 0, 1])
        w = np.concatenate((w, w), axis=1)
        self.conv2[0].weight.data.copy_(torch.from_numpy(w))
        self.conv2[0].bias.data.copy_(torch.from_numpy(b.reshape(-1)))

        w, b = data['layers'][0][8][0]['weights'][0][0]
        self.conv3[0].weight.data.copy_(torch.from_numpy(w.transpose([3, 2, 0, 1])))
        self.conv3[0].bias.data.copy_(torch.from_numpy(b.reshape(-1)))
        w, b = data['layers'][0][10][0]['weights'][0][0]
        w = w.transpose([3, 2, 0, 1])
        w = np.concatenate((w, w), axis=1)
        self.conv3[2].weight.data.copy_(torch.from_numpy(w))
        self.conv3[2].bias.data.copy_(torch.from_numpy(b.reshape(-1)))
        w, b = data['layers'][0][12][0]['weights'][0][0]
        w = w.transpose([3, 2, 0, 1])
        w = np.concatenate((w, w), axis=1)
        self.conv3[4].weight.data.copy_(torch.from_numpy(w))
        self.conv3[4].bias.data.copy_(torch.from_numpy(b.reshape(-1)))
        
        torch.nn.init.normal_(self.line[1].weight.data, mean=0, std=0.01)
        torch.nn.init.zeros_(self.line[1].bias.data)
        torch.nn.init.normal_(self.line[4].weight.data, mean=0, std=0.01)
        torch.nn.init.zeros_(self.line[4].bias.data)

    def forward(self, x, label, Iter, density):
        x = self.conv1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)
        print(x.shape)
        x = self.relu(x)
        print(x.shape)
        x=self.flatten(x)
        print(x.shape)
        x=torch.unsqueeze(x,2)
        print(x.shape)
        x=torch.unsqueeze(x,3)
        x = self.line(x)
        print(x.shape)
        return x





