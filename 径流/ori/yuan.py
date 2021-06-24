# https://zhuanlan.zhihu.com/p/94757947
import numpy as np
import torch
from torch import nn
import pandas as pd

import matplotlib.pyplot as plt

"""
Github: Yonv1943 Zen4 Jia1 hao2
https://github.com/Yonv1943/DL_RL_Zoo/blob/master/RNN
The source of training data
https://github.com/L1aoXingyu/
code-of-learn-deep-learning-with-pytorch/blob/master/
chapter5_RNN/time-series/lstm-time-series.ipynb
"""


def run_train_gru():
    inp_dim = 3
    out_dim = 1
    batch_size = 12 * 4

    """load data"""
    data = load_data()
    data_x = data[:-1, :]
    data_y = data[+1:, 0]
    assert data_x.shape[1] == inp_dim

    train_size = int(len(data_x) * 0.75)

    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    train_x = train_x.reshape((train_size, inp_dim))
    train_y = train_y.reshape((train_size, out_dim))

    """build model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegGRU(inp_dim, out_dim, mod_dim=12, mid_layers=2).to(device)
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

    for e in range(256):
        out = net(batch_var_x)

        # loss = criterion(out, batch_var_y)
        loss = (out - batch_var_y) ** 2 * weights
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 100 == 0:
            print("Epoch: {}, Loss: {:.5f}".format(e, loss.item()))

    """eval"""
    net = net.eval()

    test_x = data_x.copy()
    test_x[train_size:, 0] = 0
    test_x = test_x[:, np.newaxis, :]
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
    for i in range(train_size, len(data) - 2):
        test_y = net(test_x[:i])
        test_x[i + 1, 0, 0] = test_y[-1]
    pred_y = test_x[1:, 0, 0]
    pred_y = pred_y.cpu().data.numpy()

    diff_y = pred_y[train_size:] - data_y[train_size:-1]
    l1_loss = np.mean(np.abs(diff_y))
    l2_loss = np.mean(diff_y ** 2)
    print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))
    plt.plot(pred_y, "r", label="pred")
    plt.plot(data_y, "b", label="real")
    plt.legend(loc="best")
    plt.pause(4)


def run_train_lstm():
    inp_dim = 3  # 是LSTM输入张量的维度，我们已经根据我们的数据确定了这个值是3
    out_dim = 1  # 我们只需要预测客流量这一个值，因此out_dim 为1
    mid_dim = 8  # mid_dim 是LSTM三个门 (gate) 的网络宽度，也是LSTM输出张量的维度
    mid_layers = 1
    batch_size = 12 * 4
    mod_dir = "."

    """load data"""
    data = load_data()
    data_x = data[:-1, :]  # 删除最后一行 143*3
    data_y = data[+1:, 0]
    assert data_x.shape[1] == inp_dim

    # 取整前四分之三的训练数据数量
    train_size = int(len(data_x) * 0.75)

    train_x = data_x[:train_size]
    train_y = data_y[:train_size]  # (107,)
    train_x = train_x.reshape((train_size, inp_dim))
    train_y = train_y.reshape((train_size, out_dim))  # (107,1)
    # 以上都是准备数据
    """build model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    criterion = nn.MSELoss()
    # 优化器指定调整的参数和学习率
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    """train"""
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)

    batch_var_x = list()
    batch_var_y = list()

    for i in range(batch_size):
        j = train_size - i  # 不同的起始裁剪位点
        batch_var_x.append(var_x[j:])  # 在训练中作为输入的客流量数据（年份、月份、本月的客流量）
        batch_var_y.append(var_y[j:])  # 在训练中作为标签的预测数据（下个月的客流量）
    """
    输入多个拥有相同起始裁剪位点的序列给RNN用于训练是完全错误的训练方式。
    当你输入一个序列给RNN时，合格的深度学习框架中的RNN就已经帮你训练了拥有相同
    起始位点的所有序列，这个时候还额外地输入其他拥有相同起始裁剪位点的序列给RNN
    会导致RNN更快地过拟合。由于数据是这么地少，因此我们只能使用同一个batch进行训练
    （非常容易过拟合），但因为有这个数据构建方法，所以我们不需要训练太久，几秒钟就可以了。
    batch_var_x.append(var_x[j:])  # 不同的起始裁剪位点，正确的裁剪方法
    batch_var_x.append(var_x[:j])  # 相同的起始裁剪位点，**完全错误的裁剪方法**
     """

    from torch.nn.utils.rnn import pad_sequence

    batch_var_x = pad_sequence(batch_var_x)
    # 用于在开头添加   [0.,   0.,   0.]
    batch_var_y = pad_sequence(batch_var_y)

    with torch.no_grad():
        weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))
        weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Training Start")
    for e in range(384):
        optimizer.zero_grad()  # 梯度清零（=net.zero_grad()

        out = net(batch_var_x)
        loss = criterion(out, batch_var_y)
        # loss = (out - batch_var_y) ** 2 * weights
        loss = loss.mean()

        loss.backward()  # 反向传播

        optimizer.step()  # 更新参数

        if e % 64 == 0:
            print("Epoch: {:4}, Loss: {:.5f}".format(e, loss.item()))
    # torch.save(net.state_dict(), "{}/net.pth".format(mod_dir))
    # print("Save in:", "{}/net.pth".format(mod_dir))

    """eval"""
    # net.load_state_dict(
    #     torch.load(
    #         "{}/net.pth".format(mod_dir), map_location=lambda storage, loc: storage
    #     )
    # )
    net = net.eval()
    # 这里把上面训练好的模型保存再取出，然后划分出测试集进行训练
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
    # 我们需要保存LSTM的隐藏状态（hidden state），用于恢复序列中断后的计算。
    test_y, hc = net.output_y_hc(test_x[:train_size], (zero_ten, zero_ten))
    test_x[train_size + 1, 0, 0] = test_y[-1]
    for i in range(train_size + 1, len(data) - 2):
        test_y, hc = net.output_y_hc(test_x[i : i + 1], hc)
        test_x[i + 1, 0, 0] = test_y[-1]
    pred_y = test_x[1:, 0, 0]
    pred_y = pred_y.cpu().data.numpy()

    diff_y = pred_y[train_size:] - data_y[train_size:-1]
    l1_loss = np.mean(np.abs(diff_y))\`)|_`
    l2_loss = np.mean(diff_y ** 2)
    print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))

    plt.plot(pred_y, "r", label="pred")
    plt.plot(data_y, "b", label="real", alpha=0.3)
    plt.plot([train_size, train_size], [-1, 2], color="k", label="train | pred")
    plt.legend(loc="best")
    plt.show()
    # plt.savefig("lstm_reg.png")
    plt.pause(4)


def run_origin():
    inp_dim = 2
    out_dim = 1
    mod_dir = "."

    """load data"""
    data = load_data()  # axis1: number, year, month
    data_x = np.concatenate((data[:-2, 0:1], data[+1:-1, 0:1]), axis=1)
    data_y = data[2:, 0]

    train_size = int(len(data_x) * 0.75)
    train_x = data_x[:train_size]
    train_y = data_y[:train_size]

    train_x = train_x.reshape((-1, 1, inp_dim))
    train_y = train_y.reshape((-1, 1, out_dim))

    """build model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegLSTM(inp_dim, out_dim, mid_dim=4, mid_layers=2).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    """train"""
    var_x = torch.tensor(train_x, dtype=torch.float32, device=device)
    var_y = torch.tensor(train_y, dtype=torch.float32, device=device)
    print("var_x.size():", var_x.size())
    print("var_y.size():", var_y.size())

    for e in range(512):
        out = net(var_x)
        loss = criterion(out, var_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e + 1) % 100 == 0:  # 每 100 次输出结果
            print("Epoch: {}, Loss: {:.5f}".format(e + 1, loss.item()))

    torch.save(net.state_dict(), "{}/net.pth".format(mod_dir))

    """eval"""
    # net.load_state_dict(torch.load('{}/net.pth'.format(mod_dir), map_location=lambda storage, loc: storage))
    net = net.eval()  # 转换成测试模式

    """
    inappropriate way of seq prediction:
    use all real data to predict the number of next month
    """
    test_x = data_x.reshape((-1, 1, inp_dim))
    var_data = torch.tensor(test_x, dtype=torch.float32, device=device)
    eval_y = net(var_data)  # 测试集的预测结果
    pred_y = eval_y.view(-1).cpu().data.numpy()

    plt.plot(pred_y[1:], "r", label="pred inappr", alpha=0.3)
    plt.plot(data_y, "b", label="real", alpha=0.3)
    plt.plot([train_size, train_size], [-1, 2], label="train | pred")

    """
    appropriate way of seq prediction:
    use real+pred data to predict the number of next 3 years.
    """
    test_x = data_x.reshape((-1, 1, inp_dim))
    test_x[train_size:] = 0  # delete the data of next 3 years.
    test_x = torch.tensor(test_x, dtype=torch.float32, device=device)
    for i in range(train_size, len(data) - 2):
        test_y = net(test_x[:i])
        test_x[i, 0, 0] = test_x[i - 1, 0, 1]
        test_x[i, 0, 1] = test_y[-1, 0]
    pred_y = test_x.cpu().data.numpy()
    pred_y = pred_y[:, 0, 0]
    plt.plot(pred_y[2:], "g", label="pred appr")

    plt.legend(loc="best")
    plt.savefig("lstm_origin.png")
    plt.pause(4)


class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()
        # 默认                3        8        1
        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)
        # rnn
        # inp_dim 是LSTM输入张量的维度，我们已经根据我们的数据确定了这个值是3
        # mid_dim 是LSTM三个门 (gate) 的网络宽度，也是LSTM输出张量的维度
        # num_layers 是使用两个LSTM对数据进行预测，然后将他们的输出堆叠起来。
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


class RegGRU(nn.Module):
    def __init__(self, inp_dim, out_dim, mod_dim, mid_layers):
        super(RegGRU, self).__init__()

        self.rnn = nn.GRU(inp_dim, mod_dim, mid_layers)
        self.reg = nn.Linear(mod_dim, out_dim)

    def forward(self, x):
        x, h = self.rnn(x)  # (seq, batch, hidden)

        seq_len, batch_size, hid_dim = x.shape
        x = x.view(-1, hid_dim)
        x = self.reg(x)
        x = x.view(seq_len, batch_size, -1)
        return x

    def output_y_h(self, x, h):
        y, h = self.rnn(x, h)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, h


def load_data():
    # passengers number of international airline , 1949-01 ~ 1960-12 per month
    seq_number = np.array(
        [
            112.0,
            118.0,
            132.0,
            129.0,
            121.0,
            135.0,
            148.0,
            148.0,
            136.0,
            119.0,
            104.0,
            118.0,
            115.0,
            126.0,
            141.0,
            135.0,
            125.0,
            149.0,
            170.0,
            170.0,
            158.0,
            133.0,
            114.0,
            140.0,
            145.0,
            150.0,
            178.0,
            163.0,
            172.0,
            178.0,
            199.0,
            199.0,
            184.0,
            162.0,
            146.0,
            166.0,
            171.0,
            180.0,
            193.0,
            181.0,
            183.0,
            218.0,
            230.0,
            242.0,
            209.0,
            191.0,
            172.0,
            194.0,
            196.0,
            196.0,
            236.0,
            235.0,
            229.0,
            243.0,
            264.0,
            272.0,
            237.0,
            211.0,
            180.0,
            201.0,
            204.0,
            188.0,
            235.0,
            227.0,
            234.0,
            264.0,
            302.0,
            293.0,
            259.0,
            229.0,
            203.0,
            229.0,
            242.0,
            233.0,
            267.0,
            269.0,
            270.0,
            315.0,
            364.0,
            347.0,
            312.0,
            274.0,
            237.0,
            278.0,
            284.0,
            277.0,
            317.0,
            313.0,
            318.0,
            374.0,
            413.0,
            405.0,
            355.0,
            306.0,
            271.0,
            306.0,
            315.0,
            301.0,
            356.0,
            348.0,
            355.0,
            422.0,
            465.0,
            467.0,
            404.0,
            347.0,
            305.0,
            336.0,
            340.0,
            318.0,
            362.0,
            348.0,
            363.0,
            435.0,
            491.0,
            505.0,
            404.0,
            359.0,
            310.0,
            337.0,
            360.0,
            342.0,
            406.0,
            396.0,
            420.0,
            472.0,
            548.0,
            559.0,
            463.0,
            407.0,
            362.0,
            405.0,
            417.0,
            391.0,
            419.0,
            461.0,
            472.0,
            535.0,
            622.0,
            606.0,
            508.0,
            461.0,
            390.0,
            432.0,
        ],
        dtype=np.float32,
    )
    # assert seq_number.shape == (144, )
    # plt.plot(seq_number)
    # plt.ion()
    # plt.pause(1)
    # 新增一列
    seq_number = seq_number[:, np.newaxis]
    # print(seq_number)
    # print(repr(seq))

    # 1949~1960, 12 years, 12*12==144 month
    seq_year = np.arange(12)
    seq_month = np.arange(12)

    # numpy.repeat(a, repeats, axis=None)
    # 其中a为数组，repeats为重复的次数，axis表示数组维度
    seq_year_month = np.transpose(
        [np.repeat(seq_year, len(seq_month)), np.tile(seq_month, len(seq_year))],
    )
    # 给数据seq_number加上标号，如 0 1 xxx第0年第一月流量是xxx,最后一行为11 11 xxx
    # print(seq_year_month)

    # Cartesian Product

    seq = np.concatenate((seq_number, seq_year_month), axis=1)
    # print(seq)
    # [112.   0.   0.] 144行，每行是这样[432.  11.  11.]

    # normalization
    # 在寒假学习的ML.md中这叫做标准化
    seq = (seq - seq.mean(axis=0)) / seq.std(axis=0)
    return seq


if __name__ == "__main__":
    run_train_lstm()
    # run_train_gru()
    # run_origin()
    # print(load_data())
