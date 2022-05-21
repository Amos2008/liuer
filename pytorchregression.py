import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        #super(LinearModel, self).__init__()
        torch.nn.Module.__init__(self)
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1, 1)#同样继承torch.nn.Module  self.linear是个对象  nn neural network

    def forward(self, x):#fugai  父类forward
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()


criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # model.parameters()自动完成参数的初始化操作

# training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)  # forward:predict
    loss = criterion(y_pred, y_data)  # forward: loss
    print(epoch, loss.item())

    # optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward()  # backward: autograd，自动计算梯度
    optimizer.step()  # update 参数，即更新w和b的值

print("w=",model.linear.weight.item())
print("b=",model.linear.bias.item())

x_test =torch.Tensor([[5.0]])
y_test = model(x_test)
print("y_pred",y_test.data)


# x_data = torch.Tensor([[1.0,5,4,3,2,1,0,5], [2.0,3,2,4,5,3,5,1], [3.0,5,2,3,4,1,2,3]])
# y_data = torch.Tensor([[1], [1], [0]])
# # import torch.nn.functional as F
# class LogisticRegressionModel(torch.nn.Module):#所有均要继承module
#     def __init__(self):
#         torch.nn.Module.__init__(self)
#         # super(LogisticRegressionModel,self).__init__()#必须要有
#         self.linear1=torch.nn.Linear(8,6)
#         self.linear2=torch.nn.Linear(6,4)
#         self.linear3=torch.nn.Linear(4,1)
#         self.activate=torch.nn.Sigmoid()
#         #self.linear=torch.sigmoid(1,1)
#
#     # def __init__(self, name, age, language):  # 先继承，在重构
#     #     Person(父类）.__init__(self, name, age)  # 继承父类的构造方法，也可以写成：super(Chinese（子类）,self).__init__(name,age)
#     #     self.language = language  # 定义类的本身属性
#     def forward(self,x):
#         x=self.activate(self.linear1(x))
#         x = self.activate(self.linear2(x))
#         x = self.activate(self.linear3(x))
#         return x
#
# model = LogisticRegressionModel()
# criteration = torch.nn.BCELoss(reduction='mean')
# optimizer = torch.optim.SGD(model.parameters(),lr=0.01)#告诉优化器哪些参数，优化
#
#
# for epoch in range(1000):
#     y_pred=model(x_data)
#     loss = criteration(y_pred,y_data)
#     print(epoch,loss)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# # print("w=",model.linear.weight.item())
# # print("b=",model.linear.bias.item())
# x_test =torch.Tensor([[1.0,5,4,3,2,1,0,5]])
# y_test = model(x_test)
# print(y_test.item())






