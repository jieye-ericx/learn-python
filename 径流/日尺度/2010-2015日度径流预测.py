# https://zhuanlan.zhihu.com/p/94757947
import numpy as np
import torch
from torch import nn
import pandas as pd

import matplotlib.pyplot as plt


def run_train_lstm():
    inp_dim = 4
    out_dim = 1
    mid_dim = 8
    mid_layers = 1
    batch_size = 12 * 4
    mod_dir = "."

    """load data"""
    data = load_my_data()
    data_x = data[:-1, :]
    data_y = data[+1:, 0]
    assert data_x.shape[1] == inp_dim

    train_size = int(len(data_x) * 0.8)

    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    train_x = train_x.reshape((train_size, inp_dim))
    train_y = train_y.reshape((train_size, out_dim))

    """build model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    """train"""
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)

    batch_var_x = list()
    batch_var_y = list()

    for i in range(batch_size):
        j = train_size - i
        batch_var_x.append(var_x[j:])
        batch_var_y.append(var_y[j:])

    from torch.nn.utils.rnn import pad_sequence

    batch_var_x = pad_sequence(batch_var_x)
    batch_var_y = pad_sequence(batch_var_y)

    with torch.no_grad():
        weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Training Start")
    for e in range(200):
        out = net(batch_var_x)

        # loss = criterion(out, batch_var_y)
        loss = (out - batch_var_y) ** 2 * weights
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 64 == 0:
            print("Epoch: {:4}, Loss: {:.5f}".format(e, loss.item()))

    # torch.save(net.state_dict(), '{}/net.pth'.format(mod_dir))
    print("Save in:", "{}/net.pth".format(mod_dir))

    """eval"""
    # net.load_state_dict(torch.load('{}/net.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
    net = net.eval()

    test_x = data_x.copy()
    test_x[train_size:, 0] = 0
    test_x = test_x[:, np.newaxis, :]
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device)

    """simple way but no elegant"""
    # for i in range(train_size, len(data) - 2):
    #     test_y = net(test_x[:i])
    #     test_x[i, 0, 0] = test_y[-1]

    """elegant way but slightly complicated"""
    eval_size = 1
    zero_ten = torch.zeros(
        (mid_layers, eval_size, mid_dim), dtype=torch.float32, device=device
    )
    test_y, hc = net.output_y_hc(test_x[:train_size], (zero_ten, zero_ten))
    test_x[train_size + 1, 0, 0] = test_y[-1]
    for i in range(train_size + 1, len(data) - 2):
        test_y, hc = net.output_y_hc(test_x[i : i + 1], hc)
        test_x[i + 1, 0, 0] = test_y[-1]
    pred_y = test_x[1:, 0, 0]
    pred_y = pred_y.cpu().data.numpy()

    diff_y = pred_y[train_size:] - data_y[train_size:-1]
    l1_loss = np.mean(np.abs(diff_y))
    l2_loss = np.mean(diff_y ** 2)
    print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))

    plt.plot(pred_y, "r", label="pred")
    plt.plot(data_y, "b", label="real", alpha=0.3)
    plt.plot([train_size, train_size], [-1, 2], color="k", label="train | pred")
    plt.legend(loc="best")
    plt.savefig("lstm_1015_日尺度.png")
    plt.pause(4)


class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y

    """
    PyCharm Crtl+click nn.LSTM() jump to code of PyTorch:
    Examples::
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    def output_y_hc(self, x, hc):
        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc


def load_my_data():
    # seq_year = np.arange(12)
    # seq_month = np.arange(12)
    # seq_day_p = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # seq_day_r = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    origindata = pd.read_excel("径流/oriDataset/2010-2015.xlsx")
    data = origindata.to_numpy()
    data = data[:, [2, 0, 1]]

    newCol1 = np.zeros(len(data), dtype=np.float32)
    newCol2 = np.zeros(len(data), dtype=np.float32)
    data = np.column_stack((data, newCol1))
    data = np.column_stack((data, newCol2))
    for i in range(len(data)):
        # print(data[i][1
        data[i][1] = data[i][1].split(".")
        data[i][3] = int(data[i][1][1])
        data[i][4] = int(data[i][1][2])
        data[i][1] = int(data[i][1][0])

    data = data[:, [0, 1, 3, 4]]
    data = data.astype(dtype=np.float32)
    # 暂时舍弃第二列
    print(data.dtype)
    print(data[0])
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    # print(data[0])
    # print(origindata)
    return data


if __name__ == "__main__":
    run_train_lstm()
    # run_train_gru()
    # run_origin()
    # load_my_data()
