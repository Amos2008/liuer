import torch
print(torch.__version__)
learn_a = torch.FloatTensor(2,3)
learn_a.requires_grad = True

print("learn",learn_a)
learn_b = learn_a

print("learn_b", learn_b)
learn_c = torch.Tensor(3, 2)



learn_d = torch.add(learn_a, learn_b)
print("learn_d", learn_d)


print(learn_a.data)
print(learn_a.grad)
print(learn_d.data.type())
print(type(learn_a.grad))
print(type(learn_a))


#torch.Tensor 和torch.autograd.Variable变成同一个类了。torch.Tensor 能够像之前的Variable一样追踪历史和反向传播，
# 也即不用使用torch.autograd.Variable来封装tensor以使其能够进行反向传播。但是Variable仍能够正常工作，只是返回的依旧是Tensor。


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
#不用求梯度
w1 = torch.tensor([1.0],dtype=float,requires_grad=True)
w2 = torch.tensor([1.0],dtype=float,requires_grad=True)
b = torch.tensor([1.0],dtype=float,requires_grad=True)

#随机梯度法的w1*x**2+w2*x+b
def forward(x):
    return x*x*w1+x*w2+b
def loss(x, y):
    y_pred = forward(x)
    return (y-y_pred)**2

for epoch in range(100000):
    for x, y in zip(x_data, y_data):
        l=loss(x, y)
        l.backward()
        w1.data = w1.data -0.01*w1.grad.data
        w2.data = w2.data - 0.01*w2.grad.data
        b.data = b.data - 0.01*b.grad.data
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("progress:", epoch,"w1:",w1.item(),"w2:" ,w2.item(),"loss:",l.item())


print("predict (after training)", 4, forward(4).item())


# w = torch.Tensor([1.0])
# b = torch.Tensor([10.0])
# w.requires_grad = True
# b.requires_grad = True
#
# def forward(x):
#     return x*w + b
# def loss(x_data, y_data):
#     loss_ = 0
#     for x,y in zip(x_data, y_data):
#         y_pred = x*w +b
#         loss_ +=(y_pred -y)**2
#     return loss_
#
# for epoch in range(10):
#     loss_ = loss(x_data,y_data)
#     loss_.backward()
#     w.data = w.data - 0.05 * w.grad.data
#     b.data = b.data - 0.05* b.grad.data
#     w.grad.data.zero_()
#     b.grad.data.zero_()
#     print(w)
#     print(b)
#     print("progress:", "epoch:", epoch, "w.item:",w.item() , "b.item():",b.item(), "loss_.item():",loss_.item())

