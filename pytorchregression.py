import torch
import numpy as np
import matplotlib.pyplot as plt

# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
#
#
# class LinearModel(torch.nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
#         # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
#         self.linear = torch.nn.Linear(1, 1)#同样继承torch.nn.Module  self.linear是个对象  nn neural network
#
#     def forward(self, x):#fugai  父类forward
#         y_pred = self.linear(x)
#         return y_pred
#
#
# model = LinearModel()
#
#
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # model.parameters()自动完成参数的初始化操作
#
# # training cycle forward, backward, update
# for epoch in range(1000):
#     y_pred = model(x_data)  # forward:predict
#     loss = criterion(y_pred, y_data)  # forward: loss
#     print(epoch, loss.item())
#
#     optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
#     loss.backward()  # backward: autograd，自动计算梯度
#     optimizer.step()  # update 参数，即更新w和b的值
#
# print("w=",model.linear.weight.item())
# print("b=",model.linear.bias.item())
#
# x_test =torch.Tensor([[4.0]])
# y_test = model(x_test)
# print("y_pred",y_test.data)

x_data = torch.Tensor([[1.0], [2.0],[3.0]])
y_data = torch.Tensor([[2.0], [4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.09)

for epoch in range(1000):
    Y_pred = model(x_data)
    loss = criterion(Y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("w=", model.linear.weight.item())
print("b=", model.linear.bias.item())
