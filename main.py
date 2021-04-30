import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime as dt
import os
import numpy as np
import matplotlib.gridspec as gridspec
from pandas import DataFrame
from pandas import concat
import torch
from torch import nn
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
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
# Opzione_1: convertire l'anno in secondi

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


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1) -> X
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('temp%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n) -> Y
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('temp%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('temp%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
        return agg

"""
def scale(df, cols):
	for col in cols:
		# df[col] = df[col].astype(float)
		max = float(df[col].max())
		df[col] = df[col].apply(lambda x: x/max)
		return df

columns = df.columns[1:-5]
prova = scale(df, columns)
"""
# TODO: Normalization with the function
temp = df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)']
max_temp = df['CORE_ZN:Zone Mean Air Temperature [C](TimeStep)'].max()
temp = temp.apply(lambda x: x/max_temp)
print(temp)
data_temp = series_to_supervised(temp.to_list())
# in input le 48 ore prima
x_train = data_temp['temp1(t-1)'].iloc[-48:]
y_train = data_temp['temp1(t)'].iloc[-1:]
x_test = data_temp['temp1(t-1)'].iloc[-49:-1]
y_test = data_temp['temp1(t)'].iloc[-2:-1]


x_train = torch.tensor(np.array(x_train.values)).float()
y_train = torch.tensor(np.array(y_train.values)).float()
x_test = torch.tensor(np.array(x_test.values)).float()
y_test = torch.tensor(np.array(y_test.values)).float()


# Model definition
class MLP(nn.Module):
	# define model elements
	def __init__(self):
		super(MLP, self).__init__()
		self.hidden1 = Linear(48, 10) # input to first hidden layer
		self.act1 = ReLU()
		self.hidden2 = Linear(10, 8) # second hidden layer
		self.act2 = ReLU()
		self.hidden3 = Linear(8, 1) # third hidden layer and output
		self.act3 = ReLU()

	# forward propagate input
	def forward(self, X):
		# input to first hidden layer
		X = self.hidden1(X)
		X = self.act1(X)
		# second hidden layer
		X = self.hidden2(X)
		X = self.act2(X)
		# third hidden layer and output
		X = self.hidden3(X)
		X = self.act3(X)
		return X


train_dl = DataLoader(x_train, batch_size=12, shuffle=False)
test_dl = DataLoader(x_test, batch_size=12, shuffle=False)

mlp = MLP()
optimizer = optim.Adam(mlp.parameters(), lr=0.01) # lr "learning rate"

#training with multiple epochs
for epoch in range(4):
    total_loss = 0
    total_correct = 0
    for batch in train_dl: # get batch
        # images, labels = batch
        preds = mlp(x_train) # pass batch
        loss = F.cross_entropy(preds, y_train) # calculate the loss

        optimizer.zero_grad()
        loss.backward() #calculate the gradient
        optimizer.step() # update weight

        total_loss += loss.item()
        # total_correct += get_num_correct(preds, labels)

	print("epoch: ", epoch, "loss: ", total_loss)
    # print("epoch: ", epoch, "total_correct: ", total_correct, "loss: ", total_loss)

# total_correct/len(train_set)



