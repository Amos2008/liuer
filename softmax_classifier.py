
# import numpy as np
# import torch
#
# y = np.array([1, 0, 0])
# z = np.array([0.2, 0.1, -0.1])
# y_pred = np.exp(z) / np.exp(z).sum()
# loss = (- y * np.log(y_pred)).sum()
# print(loss)


# import torch
# criterion = torch.nn.CrossEntropyLoss()
# Y = torch.LongTensor([2, 0, 1])
# Y_pred1 = torch.Tensor([[0.1, 0.2, 0.9],
# [1.1, 0.1, 0.2],
# [0.2, 2.1, 0.1]])
# Y_pred2 = torch.Tensor([[0.8, 0.2, 0.3],
# [0.2, 0.3, 0.5],
# [0.2, 0.2, 0.5]])
# l1 = criterion(Y_pred1, Y)
# l2 = criterion(Y_pred2, Y)
# print("Batch Loss1 = ", l1.data, "\nBatch Loss2=", l2.data)

# import torch
# from torchvision import transforms
# from torchvision import datasets
# from torch.utils.data import DataLoader
# import torch.nn.functional as F#用relu  不用sigmoid
# import torch.optim as optim
# import cv2
#
# batch_size = 64
# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307, ), (0.3081, ))])
#
# train_dataset = datasets.MNIST(root='../dataset/mnist/',train=True,download=True,transform=transform)
#
# train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
#
# test_dataset = datasets.MNIST(root='../dataset/mnist/',train=False,download=True,transform=transform)
#
# test_loader = DataLoader(test_dataset,shuffle=False,batch_size=batch_size)
#
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.l1 = torch.nn.Linear(784, 512)
#         self.l2 = torch.nn.Linear(512, 256)
#         self.l3 = torch.nn.Linear(256, 128)
#         self.l4 = torch.nn.Linear(128, 64)
#         self.l5 = torch.nn.Linear(64, 10)
#     def forward(self, x):
#         x = x.view(-1, 784)
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = F.relu(self.l3(x))
#         x = F.relu(self.l4(x))
#         return self.l5(x)
#
# model = Net()
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#
# def train(epoch):
#     running_loss = 0.0
#     for batch_idx, data in enumerate(train_loader, 0):
#         inputs, target = data
#         optimizer.zero_grad()
#         # forward + backward + update
#         outputs = model(inputs)
#         loss = criterion(outputs, target)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if batch_idx % 300 == 299:
#             print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
#             running_loss = 0.0
#
#
# def test():
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             outputs = model(images)
#             # Image = images[0].numpy()
#             #
#             # cv2.imshow("Image",Image)
#             # cv2.waitKey(0)
#             # _, predicted = torch.max(outputs.data, dim=1)
#             _, predicted = torch.max(outputs.data, dim=1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     print('Accuracy on test set: %d %%' % (100 * correct / total))
#
# if __name__ == '__main__':
#     for epoch in range(10):
#         train(epoch)
#         test()

import torch
from torchvision import transforms  # 对图像进行处理的工具
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 使用激活函数relu()的包
import torch.optim as optim  # 优化器的包


batch_size = 64
# 对图像进行预处理，将图像转换为
transform = transforms.Compose([
    # 将原始图像PIL变为张量tensor(H*W*C),再将[0,255]区间转换为[0.1,1.0]
    transforms.ToTensor(),
    # 使用均值和标准差对张量图像进行归一化
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        #super(Net, self).__init__()
        torch.nn.Module.__init__(self)
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # 改变形状，相当于numpy的reshape
        # view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变。
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)


model = Net()
# 交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
# model.parameters()直接使用的模型的所有参数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # momentum动量

epoch = 10
def train():
    running_loss = 0.0
    # 返回了数据下标和数据
    for batch_idx, data in enumerate(train_loader, 0):
        # 送入两个张量，一个张量是64个图像的特征，一个张量图片对应的数字  0表示从零开始
        inputs, target = data
        # 梯度归零
        optimizer.zero_grad()
        # forward+backward+update
        outputs = model(inputs)
        # 计算损失，用的交叉熵损失函数
        loss = criterion(outputs, target)
        # 反馈
        loss.backward()
        # 随机梯度下降更新
        optimizer.step()

        # 每300次输出一次
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("batch_idw:",batch_idx)
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

train()
def test():
    correct = 0
    total = 0
    # 不会计算梯度
    with torch.no_grad():
        for data in test_loader:  # 拿数据
            images, labels = data
            outputs = model(images)  # 预测
            # outputs.data是一个矩阵，每一行10个量，最大值的下标就是预测值
            _, predicted = torch.max(outputs.data, dim=1)  # 沿着第一维度，找最大值的下标，返回最大值和下标
            total += labels.size(0)  # labels.size(0)=64 每个都是64个元素，就可以计算总的元素
            # (predicted == labels).sum()这个是张量，而加了item()变为一个数字，即相等的数量
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))  # 正确的数量除以总数




if __name__ == '__main__':
    # for epoch in range(1000):
        train(1)
        test()
