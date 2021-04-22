import torch
import numpy as np

# print(torch.ones(3,4,dtype=torch.int32))#生成全是1的数组
# print(torch.empty(5,3))# 生成无序数数组
# print(torch.rand(5,3))# 生成0-1随机数组
# print(torch.zeros(5,3,dtype=torch.long))# 生成0数组
# print(torch.zeros(5,3).long())# 生成0数组
# x=torch.rand(5,3)
# x=torch.rand_like(x,dtype=torch.double)

# y=torch.rand(5,3)
# print(x+y)
# print(torch.add(x,y))
# print(y.add_(x))# y本身发生变化

# print(torch.cuda.is_available())# 呼唤英伟达gpu

# Numpy实现
# N = 64  # 训练数据个数
# D_in = 1000  # 输入维度
# H = 100  # 隐藏层神经元个数？
# D_out = 10  # 输出维度
#
# # 随机训练数据
# x = np.random.randn(N, D_in)
# y = np.random.randn(N, D_out)
#
# w1 = np.random.randn(D_in, H)
# w2 = np.random.randn(H, D_out)
#
# learning_rate = 1e-6
#
# for it in range(500):
#     h = x.dot(w1)  # N*H
#     h_relu = np.maximum(h, 0)  # N*H
#     y_pred = h_relu.dot(w2)  # N*D_out
#
#     #     计算损失
#     loss = np.square(y_pred - y).sum()  # 均方误差
#     print(it, loss)
#     # 计算梯度，手动
#     grad_y_pred=2.0*(y_pred-y)
#     grad_w2=h_relu.T.dot(grad_y_pred)
#     grad_h_relu=grad_y_pred.dot(w2.T)
#     grad_h=grad_h_relu.copy()
#     grad_h[h<0]=0
#     grad_w1=x.T.dot(grad_h)
#
#     # 更新w1,w2
#     w1-=learning_rate*grad_w1
#     w2-=learning_rate*grad_w2

# torch实现
# N = 64  # 训练数据个数
# D_in = 1000  # 输入维度
# H = 100  # 隐藏层神经元个数？
# D_out = 10  # 输出维度
#
# # 随机训练数据
# x = torch.randn(N, D_in)
# y = torch.randn(N, D_out)
# 随机初始化权重
# w1 = torch.randn(D_in, H, requires_grad=True)
# w2 = torch.randn(H, D_out, requires_grad=True)
#
# learning_rate = 1e-6
#
# for it in range(500):
# 前向传播：计算预测值y
#     # h = x.mm(w1)  # N*H
#     # h_relu = h.clamp(min=0)  # N*H
#     y_pred = x.mm(w1).clamp(min=0).mm(w2)  # N*D_out
#
#     #     计算损失
#     loss = (y_pred - y).pow(2).sum()  # 均方误差,代表一个计算图
#     print(it, loss)
#     loss.backward()
#
#     # 更新w1,w2
#     with torch.no_grad():
#         w1 -= learning_rate * w1.grad
#         w2 -= learning_rate * w2.grad
#         w1.grad.zero_()
#         w2.grad.zero_()


# --------------------nn
print(torch.cuda.is_available())

N = 64  # 训练数据个数
D_in = 1000  # 输入维度
H = 100  # 隐藏层维度
D_out = 10  # 输出维度

# 随机训练数据
x = torch.randn(N, D_in)  # 64行1000列
y = torch.randn(N, D_out)  # 64行10列

# for it in range(500):
#     y_pred = model(x)  # model.forward()

#     #     计算损失

#     loss = loss_fn(y_pred, y)  # 均方误差,代表一个计算图
#     print(it, loss.item())

#     optimizer.zero_grad()
#     loss.backward()

#     optimizer.step()
