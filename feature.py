import torch
import numpy as np
import matplotlib.pyplot as plt
# epoch 所有训练样本均训练   mini-batch每次训练的样本数量  iterations 内层的迭代多少次
# DataLoader  batch_size   shuffle每次打乱顺序   先shuffle  再batch_size

# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])


x_data = torch.Tensor([[1.0,5,4,3,2,1,0,5], [2.0,3,2,4,5,3,5,1], [3.0,5,2,3,4,1,2,3]])
y_data = torch.Tensor([[1], [1], [0]])
# import torch.nn.functional as F
class LogisticRegressionModel(torch.nn.Module):#所有均要继承module
    def __init__(self):
        super(LogisticRegressionModel,self).__init__()#必须要有
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.activate=torch.nn.Sigmoid()
        #self.linear=torch.sigmoid(1,1)

    def forward(self,x):
        x=self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x

model = LogisticRegressionModel()
criteration = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)#告诉优化器哪些参数，优化


for epoch in range(100000):
    y_pred=model(x_data)
    loss = criteration(y_pred,y_data)
    print(epoch,loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# print("w=",model.linear.weight.item())
# print("b=",model.linear.bias.item())
x_test =torch.Tensor([[1.0,5,4,3,2,1,0,5]])
y_test = model(x_test)
print(y_test.item())