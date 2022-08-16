import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data=torch.from_numpy(xy[:,:-1])
        self.y_data=torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

aa = DiabetesDataset('diabetes.csv')
print(aa.len)
print(aa.x_data)
print(aa.y_data)

dataset = DiabetesDataset('diabetes.csv.gz')
train_loader =DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)

testset =DiabetesDataset('diabetes.csv.gz')
test_loader =DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)

class Model(torch.nn.Module):#所有均要继承module
    def __init__(self):
        # super(Model,self).__init__()#必须要有
        torch.nn.Module.__init__(self)
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()
        #self.linear=torch.sigmoid(1,1)

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(20):
        for i,data in enumerate(train_loader,0):
            inputs,labels=data
            y_pred =model(inputs)
            loss = criterion(y_pred,labels)
            print(epoch,i,loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# 便携式水位流速仪合同细节内部讨论及确定 包括项目结点、项目阶段性汇报及交流，验收标准
# 与第三方瑞纳明科技沟通硬件的安装、组装方式，及前期设计实现细节
# 便携式水位流速仪室外场景测试的测试方案（solidworks简易建模并初步实现方案）
#
# 便携式水位流速仪合同确定，经审议后提交采购流程
# 室外场景测试方案细节确定提交采购