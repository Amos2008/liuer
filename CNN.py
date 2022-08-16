import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

#prepare dataset

batch_size =64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

#design model using class

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)#默认padding为2
        self.fc = torch.nn.Linear(320,10)


    def forward(self,x):
        batch_size= x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size,-1)
        x = self.fc(x)
        return x

model = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "CPU")
#cuda:0 表示同一块显卡
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01, momentum=0.5)

#training cycle forward , backward , updata

def train(epoch):
    running_loss =0.0
    for batch_idx, data in enumerate(train_loader,0):
        inputs, target =data
        inputs,targe = inputs.to(device),target.to(device)
        optimizer.zero_grad()

        outputs =model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)
        test()




























# in_channels, out_channels = 5, 10
# width, height = 100, 100
# kernel_size = 3
# batch_size = 1
#
# input = torch.randn(batch_size, in_channels, width, height)
#
# conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size =kernel_size)
#
# output = conv_layer(input)
#
#
# print(input.shape)
# print(output.shape)
# print(conv_layer.weight.shape)
#
# input = [1,2,3,4,5,
#          1,2,3,4,5,
#          1,2,3,4,5,
#          1,2,3,4,5,
#          1,2,3,4,5,]
#
# input = torch.Tensor(input).view(1,1,5,5)
# conv_layer = torch.nn.Conv2d(1, 1, kernel_size =3,padding=1,stride=2, bias= False )
# kernel = torch.Tensor([1, 2, 3, 4, 5 ,6 , 7, 8 ,9]).view(1,1,3,3)
# conv_layer.weight.data =kernel.data
# output = conv_layer(input)
#
# print(output)




