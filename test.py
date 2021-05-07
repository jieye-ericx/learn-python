import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

data = np.arange(1, 17)
# print(data)
data_x = data[:-1]
data_y = data[+1:]
# print(data_x)
# print(data_y)
train_size = 12

train_x = data_x[:train_size]
train_y = data_y[:train_size]
# print(train_x)
# print(train_y)
# 1*12的矩阵变成12*1
train_x = train_x.reshape((train_size, 1))
train_y = train_y.reshape((train_size, 1))
# print(train_x)
# print(train_y)

# list转换成tensor
# var_y = train_y
# var_x = train_x
var_y = torch.tensor(train_y, dtype=torch.float32)
var_x = torch.tensor(train_x, dtype=torch.float32)

batch_var_x = list()
batch_var_y = list()

batch_size = 6
for i in range(batch_size):
    j = train_size - i
    # print(j)
    # j=12,11,10,9,8,7
    batch_var_x.append(var_x[j:])
    batch_var_y.append(var_y[j:])

# print(batch_var_x)
# print(batch_var_y)
# print(batch_var_x.shape)
batch_var_x = pad_sequence(batch_var_x)
batch_var_y = pad_sequence(batch_var_y)
print(batch_var_x.shape)
print(batch_var_x)

# print(batch_var_y)
