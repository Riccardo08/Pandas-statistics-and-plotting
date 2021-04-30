import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from datetime import datetime as dt
import os
import numpy as np
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torchvision #provides acces to popular datasets, model architextures and image transformation for computer vision
import torchvision.transforms as transforms # provides common transformation for image processing
import torch.nn.functional as F
import torch.optim as optim

df = pd.read_csv('ASHRAE90.1_OfficeSmall_STD2016_NewYork.csv')
df=df.iloc[288:,:]
Date = df['Date/Time'].str.split(' ', expand=True)
Date.rename(columns={0:'nullo',1:'date',2:'null', 3:'time'},inplace=True)
Date['time'] = Date['time'].replace(to_replace='24:00:00', value= '0:00:00')
data = Date['date']+' '+Date['time']
data = pd.to_datetime(data, format = '%m/%d %H:%M:%S')

df['day']=data.apply(lambda x: x.day)
df['month']= data.apply(lambda x: x.month)
df['hour']=data.apply(lambda x: x.hour)
df['dn']=data.apply(lambda x: x.weekday())
df['data']=Date.date

def mean_plus_std(x):
    return x.mean()+x.std()
def mean_minus_std(x):
    return x.mean()-x.std()

def fill_dataset(df, value):
    if value == 'hour':
        new_df = df.groupby(df.hour)[df.columns[1:-5]].agg(['mean', 'std', 'max', 'min', 'median', mean_plus_std, mean_minus_std])
    elif value == 'day':
        new_df = df.groupby(df.dn)[df.columns[1:-5]].agg(['mean', 'std', 'max', 'min', 'median', mean_plus_std, mean_minus_std])
    elif value == 'month':
        new_df = df.groupby(df.month)[df.columns[1:-5]].agg(['mean', 'std', 'max', 'min', 'median', mean_plus_std, mean_minus_std])

    return new_df

df_hourly = fill_dataset(df,'hour')
df_daily = fill_dataset(df,'day')
df_monthly = fill_dataset(df,'month')


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) # 2 convolutional layers
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120) # 3 linear layers
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t
        # (2) hidden con layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)  # perchè il convolutional layer da in output solo un vettore
        t = self.fc1(t)
        t = F.relu(t)
        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        # t = F.softmax(t, dim=1) # the SOFTMAX activation function returns the probability for each of the prediction calsses
        return t


"""
train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
"""

# definire train e test

train_set = torch.utils.data.DataLoader(train, batch_size =100)
test_set = torch.utils.data.DataLoader(test, batch_size =100)


network = Network()
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01) # lr "learning rate"

for epoch in range(5):
    total_loss = 0
    total_correct = 0
    for batch in train_loader: # get batch
        images, labels = batch
        preds = network(images) # pass batch
        loss = F.cross_entropy(preds, labels) # calculate the loss

        optimizer.zero_grad()
        loss.backward() #calculate the gradient
        optimizer.step() # update weight

        total_loss += loss.item()
        total_correct += get_num_correct(preds, labels)

    print("epoch: ", epoch, "total_correct: ", total_correct, "loss: ", total_loss)

total_correct/len(train_set)

