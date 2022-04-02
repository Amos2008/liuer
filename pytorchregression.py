import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([1.0], [2.0], [3.0])
y_data = torch.Tensor([2.0], [4.0], [6.0])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()


criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # model.parameters()自动完成参数的初始化操作

# training cycle forward, backward, update
for epoch in range(100):
    y_pred = model(x_data)  # forward:predict
    loss = criterion(y_pred, y_data)  # forward: loss
    print(epoch, loss.item())

    optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward()  # backward: autograd，自动计算梯度
    optimizer.step()  # update 参数，即更新w和b的值

# x_data = [1.0, 2.0, 3.0]
# y_data = [2.0, 4.0, 6.0]
# w = torch.Tensor([1.0])
# w.requires_grad = True
# def forward(x):
#     return x*w
#
# def loss(x, y):
#     y_pred = forward(x)
#     return (y_pred -y)**2
#
#
# for epoch in range(100):
#     for x, y in zip(x_data, y_data):
#         l = loss(x,y)
#         l.backward()
#         w.data=w.data - 0.01*w.grad.data
#         w.grad.data.zero_()
#     print("progress:", epoch, l.item())