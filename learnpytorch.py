import torch

print(torch.__version__)
a = torch.tensor([1, 2])
print(a)
print(a.type())
print(a.data)
print(a.data.type())
print(a.grad)
print(type(a.grad))

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# #不用求梯度
# w = torch.Tensor([1.0])
# w.requires_grad = True
# b = torch.Tensor([1.0])
# b.requires_grad = True
#
# def forward(x):
#     return x*w
# def loss(x, y):
#     y_pred = forward(x)
#     return (y-y_pred)**2
#
# for epoch in range(100):
#     for x, y in zip(x_data, y_data):
#         l=loss(x, y)
#         l.backward()
#         w.data = w.data -0.01*w.grad.data
#         b.data = b.data - 0.01.b.grad.data
#         w.grad.data.zero_()
#         b.grad.data.zero_()
#     print("progress:", epoch, l.item())
#
#
# print("predict (after training)", 4, forward(4).item())


w = torch.Tensor([1.0])
b = torch.Tensor([10.0])
w.requires_grad = True
b.requires_grad = True

def forward(x):
    return x*w + b
def loss(x_data, y_data):
    loss_ = 0
    for x,y in zip(x_data, y_data):
        y_pred = x*w +b
        loss_ +=(y_pred -y)**2
    return loss_

for epoch in range(100000):
    loss_ = loss(x_data,y_data)
    loss_.backward()
    w.data = w.data - 0.01 * w.grad.data
    b.data = b.data - 0.01* b.grad.data
    w.grad.data.zero_()
    b.grad.data.zero_()
    print(w)
    print(b)
    print("progress:", epoch, loss_.item())