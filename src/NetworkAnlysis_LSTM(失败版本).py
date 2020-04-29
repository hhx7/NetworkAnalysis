import pandas as pd
from torch.nn import Sequential
from torch.nn import Conv2d
from torch.nn import Conv1d
from torch.nn import ReLU
from torch.nn import BatchNorm1d
from torch.nn import Softmax
from torch.nn import Sigmoid
from torch.nn import Tanh
from torch.nn import AdaptiveAvgPool2d, MaxPool2d
from torch.nn import MaxPool1d
from torch.nn import Dropout
from torch.nn import Module
from torch.nn import Linear
from torch.nn import LSTM
from torch.nn import LSTMCell
from torch.nn.init import orthogonal_
from torch.autograd import Variable
from torch import optim
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import torchvision.transforms as transforms
import torchvision.datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import normalize
import os
import torch.nn.functional as F
import math

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot

from sklearn.preprocessing import normalize
base_dataset_dir = '/home/a/Downloads/NetworkAnalysis/dataset/'
dataset_dir = '/home/a/Downloads/NetworkAnalysis/dataset/data'
out_dir = '/home/a/Downloads/NetworkAnalysis/dataset/data_new/'

batch_size = 100  #
num_classes = 1 #50
epochs = 300  #
learning_rate = 0.001
test_size = 0.2

#bilstm
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 3
drop_prob = 0#0.3
bidirectional = False
weight_decay = 0



history = {
    'loss': [],
    'val_loss': [],
    'acc': [],
    'val_acc': [],
    'max_val_acc': 0,
    'best_model': None
}
headers = ['index', 'time_diff', 'scr_ip', 'dst_ip', 'protocol',
           'length', 'content', 'seq', 'ack', 'retrans_flag',
           'wsize', 'rtt', 'uncertain_bytes', 'tcp_flag']
removed_feature = ['scr_ip', 'dst_ip',  'content', 'retrans_flag']


# 数据预处理
# 读入数据
# home, dirs, files = list(os.walk(dataset_dir))[0]
# for file_name in files:
#     # 初始化header
#     data = pd.read_csv(os.path.join(home, file_name), names=headers)
#     # 填充0
#     data.fillna(0, inplace=True)
#     # 添加属性direction
#     # #客户端-> 公网： 0
#     data['direction'] = data['dst_ip'].map(lambda ip: 1 if ip == '1.1.1.1' else 0)
#     # 转换属性protocol, tcp_flag,one-hot编码
#     data['protocol'] = data['protocol'].map({
#         'TCP': 0,
#         'TLSv1': 1,
#         'TLSv1.3': 2,
#         'SSLv2': 3
#     })
#     data['tcp_flag'] = data['tcp_flag'].map({
#         '0x00000002': 0,
#         '0x00000012': 1,
#         '0x00000010': 2,
#         '0x00000018': 3,
#         '0x00000011': 4,
#         '0x00000004': 5
#     })
#     # 消除属性index, scr_ip, dst_ip, protocol, content, retrans_flag, tcp_flag
#     data.drop(removed_feature, axis=1, inplace=True)
#     data.to_csv(out_dir + file_name, mode='w', index=False, header=True)

#
# #solution 1
home, dirs, files = list(os.walk(out_dir))[0]
X = []
y = []
lengths = []
max_rows_num = 0
feature_num = 0
avg_rows_num = 0
rows_num = 0

count_rows_freq = {}
for file_name in files:
    # 获取lable
    label = file_name[0: file_name.find('-')]
    if int(label) in range(num_classes):
        data = pd.read_csv(os.path.join(home, file_name))
        data_list = data.values.tolist()
        rows_num += len(data_list)
        if len(data_list) in count_rows_freq:
            count_rows_freq[len(data_list)] += 1
        else:
            count_rows_freq[len(data_list)] = 1

        max_rows_num = max(max_rows_num, len(data_list))
        X.append(data_list)
        y.append(int(label))
feature_num = len(X[0][0])
avg_rows_num = math.floor(rows_num / len(X))
print(max_rows_num)
print(len(X))
input_size = feature_num
sequence_length = avg_rows_num

# 填充图像
def padding(dataset, critical_val, feature_num):
    for data  in dataset:
        if critical_val <= len(data):
            # 截取
            for i in range(len(data), critical_val, -1):
                data.pop(i-1)
        else:
            # 填充0
            for i in range(len(data), critical_val):
                data.append([0] * feature_num)

# padding(X, avg_rows_num, feature_num)

# # 填充图像
# # for data in X:
# #     # 填充0
# #     for i in range(len(data), max_rows_num):
# #         data.append([0] * feature_num)
#
#
# #solution 2
# # data = pd.read_csv(base_dataset_dir + 'data_single.csv', skipinitialspace=True)
# # X = data[data.columns.difference(['label'])].values
# # y = data['label'].values
# # feature_num = len(X[0])
#
#
#
# X, y = map(
#     torch.FloatTensor, (X,y)
# )

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state = 42)
X_train_lengths = torch.tensor([len(v) for v in X_train])
X_test_lengths =  torch.tensor([len(v) for v in X_test])

# print(avg_rows_num)
# for key, value in count_rows_freq.items():
#     print('[' + str(key) + ', ' + str(value) + '],')


padding(X_train, max_rows_num, feature_num)
padding(X_test, max_rows_num, feature_num)
    # normalize
    # data = normalize(data, norm='l2')
X_train_lengths, X_train_indices = torch.sort(X_train_lengths, descending=True)
X_test_lengths, X_test_indices = torch.sort(X_test_lengths, descending=True)
X_train = torch.FloatTensor(X_train)[X_train_indices]
Y_train = torch.LongTensor(Y_train)[X_train_indices]
X_test = torch.FloatTensor(X_test)[X_test_indices]
Y_test = torch.LongTensor(Y_test)[X_test_indices]
# X_train, Y_train, X_test, Y_test = map(
#     torch.FloatTensor, (X_train, Y_train, X_test, Y_test)
# )
X_train = pack_padded_sequence(X_train, X_train_lengths, batch_first=True, enforce_sorted=True)
X_test = pack_padded_sequence(X_test, X_test_lengths, batch_first=True, enforce_sorted=True)
print(X_train.data.size())


########################test###################################

# # MNIST Dataset
# train_dataset = dsets.MNIST(root='./data/',
#                              train=True,
#                              transform=transforms.ToTensor(),
#                              download=True)
# test_dataset = dsets.MNIST(root='./data/',
#                             train=False,
#                             transform=transforms.ToTensor())
#
#  # Data Loader (Input Pipeline)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                             batch_size=batch_size,
#                                             shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=False)


##############################################################


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Lambda(Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x, y):
    #cnn
    # return x.view(len(x), 1, avg_rows_num, feature_num).to(device), y.to(device)
    #cnn1d
    # return x.view(len(x), feature_num, avg_rows_num).to(device), y.to(device)
    #rnn
    # return x.view(len(x), sequence_length, input_size).to(device), y.to(device)
    #variable length rnn
    lengths = torch.tensor([len(v) for v in x])
    lengths, indices = torch.sort(lengths, descending=True)
    padding(x, max_rows_num, feature_num)
    x, y = torch.FloatTensor(x), torch.LongTensor(y)
    x, y = x[indices],  y[indices]
    x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
    return x.to(device), y.to(device)
    #test
    # return x.view(-1, 28, 28).cuda(), y.cuda()
class MyDataset(Dataset):
    def __init__(self, *data):
        super(MyDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        return tuple(v[index] for v in self.data)

    def __len__(self):
        return len(self.data[0])

def collate_fn(train_data):
    train_data.sort(key=lambda data: len(data), reverse=True)
    data_length = [len(data) for data in train_data]
    train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)
    return train_data, data_length

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=False),
        DataLoader(valid_ds, batch_size=bs * 2),
    )



class SPPLayer(Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


class CNN_SPP(Module):
    def __init__(self, spp_level=3, device='cpu'):
        super(CNN_SPP, self).__init__()
        self.spp_level = spp_level
        self.num_grids = 0
        self.device = device
        for i in range(spp_level):
            self.num_grids += 2 ** (i * 2)

        self.conv_model = Sequential(
            Conv2d(in_channels=1, out_channels=128, padding=1, kernel_size=3),
            ReLU(),
            # MaxPool2d(2),
            Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3),
            ReLU(),
            MaxPool2d(2),
            ReLU(),
            # MaxPool2d(2),
            # Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            # ReLU()
        )

        self.spp_level = SPPLayer(spp_level)

        self.linear_model = Sequential(
            Linear(self.num_grids * 128, 1024),
            ReLU(),
            # Dropout(0.5),
            Linear(1024, 50),
            Softmax()
        )

    def forward(self, x):
        self.to(self.device)
        x = self.conv_model(x)
        x = self.spp_level(x)
        x = self.linear_model(x)
        return x





class CNN(Module):
    def __init__(self, num_classes, device='cpu'):
        super(CNN, self).__init__()
        self.device = device
        self.conv_model = Sequential(
            Conv2d(in_channels=1, out_channels=64, padding=1, kernel_size=3),
            ReLU(),
            #MaxPool2d(2),
            Conv2d(in_channels=64, out_channels=32,padding=0,  kernel_size=3),
            ReLU(),
            MaxPool2d(kernel_size=3, padding=0),

        )
        self.linear_model = Sequential(
            Lambda(lambda x: x.view(x.size(0), -1)),
            Lambda(lambda x: Linear(x.size(1), 1000).to(self.device).forward(x)),
            # ReLU(),
            # Dropout(0.5),
            # Linear(10000, 5000),
            # ReLU(),
            # Dropout(0.5),
            # Linear(5000, 1000),
            # ReLU(),
            # Dropout(0.5),
            Linear(1000, 128),
            ReLU(),
            Dropout(0.5),
            Linear(128, num_classes),
            Softmax(dim=1)
        )

    def forward(self, x):
        self.to(self.device)
        x = self.conv_model(x)
        x = self.linear_model(x)
        return x

class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers, drop_prob, num_classes,bidirectional, device='cpu'):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device
        self.factor = 2 if self.bidirectional else 1
        self.lstm = LSTM(input_size, hidden_size, num_layers, dropout=drop_prob, bidirectional=bidirectional, batch_first=True)
        # orthogonal_(self.lstm.all_weights[0][0])
        # orthogonal_(self.lstm.all_weights[0][1])
        self.fc = Sequential(
            # Dropout(0.5),
            # BatchNorm1d(hidden_size * self.factor),
            # Linear(hidden_size, 64),
            # ReLU(),
            # BatchNorm1d(64),
            Linear(hidden_size * self.factor, num_classes)
            # Softmax(dim=1)
            # Sigmoid()
        )

    def forward(self, x):
        self.to(self.device)
        hidden = self.initHidden(x)
        x, (h_n, c_n) = self.lstm(x, hidden)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x.contiguous()

        x = self.fc(x[:, -1, :])
        return x

    def initHidden(self, x):

        return torch.rand(self.num_layers * self.factor, x.data.size(0), self.hidden_size).to(self.device), torch.rand(self.num_layers * self.factor, x.data.size(0), self.hidden_size).to(self.device)

class CNN1d(Module):
    def __init__(self, num_classes, device='cpu'):
        super(CNN1d, self).__init__()
        self.device = device
        self.conv_model = Sequential(
            Conv1d(in_channels=11, out_channels=32, kernel_size=3),
            ReLU(),
            Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            ReLU(),
            MaxPool1d(2),
            Dropout(0.25)
        )

        self.linear_model = Sequential(
                Lambda(lambda x: x.view(x.size(0), -1)),
                Lambda(lambda x: Linear(x.size(1), 128).to(self.device).forward(x)),
                ReLU(),
                Dropout(0.5),
                Linear(128, num_classes),
                Softmax(dim=1)
        )
    def forward(self, x):
        self.to(self.device)
        x = self.conv_model(x)
        x = self.linear_model(x)
        return x



# model = CNN_SPP(3, device)
# model = CNN(num_classes, device)
model = RNN(input_size, hidden_size,  num_layers, drop_prob, num_classes, bidirectional, device)
# model = CNN1d(num_classes, device)
test_flag = False


def loss_batch(model, loss_func, xb, yb, opt=None):
    out = model(xb)
    _, pred = torch.max(out, 1)
    num_correct = (pred == yb).sum().item()
    # if test_flag:
    #     # print(out)
    #     print(pred)
    #     print(yb)

    loss = loss_func(out, yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb), num_correct


# def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
def fit(epochs, model, loss_func, opt, x, y, xb, yb):
    for epoch in range(epochs):
        model.train()
        # for xb, yb in train_dl:
        #     train_losses, train_nums, train_num_correct = loss_batch(model, loss_func, xb, yb, opt)

        train_losses, train_nums, train_num_correct = loss_batch(model, loss_func, x, y, opt)

        model.eval()
        global test_flag
        test_flag = True
        with torch.no_grad():
        #     valid_losses, valid_nums, valid_num_correct = zip(
        #         *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
        #     )
            valid_losses, valid_nums, valid_num_correct = loss_batch(model, loss_func, xb, yb)

        loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        acc = np.sum(train_num_correct) / np.sum(train_nums)
        val_loss = np.sum(np.multiply(valid_losses, valid_nums)) / np.sum(valid_nums)
        val_acc = np.sum(valid_num_correct) / np.sum(valid_nums)

        #record history
        history['loss'].append(loss)
        history['acc'].append(acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['best_model'] = model if history['max_val_acc'] < val_acc else history['best_model']
        history['max_val_acc'] = max(history['max_val_acc'], val_acc)

        print('epoch:' + str(epoch))
        print('train_loss: {:.6f}'.format(loss))
        print('train_acc: {:.6f}'.format(acc))
        print('valid_loss: {:.6f}'.format(val_loss))
        print('valid_acc: {:.6f}'.format(val_acc))
        test_flag = False


opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
loss_func = CrossEntropyLoss()


# train_ds = TensorDataset(X_train, Y_train)
# valid_ds = TensorDataset(X_test, Y_test)
# train_ds = MyDataset(X_train, Y_train)
# valid_ds = MyDataset(X_test, Y_test)
# train_dl, valid_dl = get_data(train_ds, valid_ds, batch_size)
# train_dl = WrappedDataLoader(train_dl, preprocess)
# valid_dl = WrappedDataLoader(valid_dl, preprocess)
# train_dl = WrappedDataLoader( train_loader, preprocess)
# valid_dl = WrappedDataLoader(test_loader,  preprocess)
# fit(epochs, model, loss_func, opt, train_dl, valid_dl)
fit(epochs, model, loss_func, opt, X_train.to(device), Y_train.to(device),  X_test.to(device), Y_test.to(device))

#draw history
pyplot.plot(history['loss'])
pyplot.plot(history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend( ['train', 'validation'], loc='upper right')
pyplot.savefig('./loss.png')
print('max_val_loss: {:.6f}'.format(history['max_val_acc']))

# 保存
torch.save(history['best_model'], 'model.pkl')
# 加载
# model = torch.load('\model.pkl')
