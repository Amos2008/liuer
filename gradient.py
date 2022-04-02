import numpy as np
import matplotlib.pyplot as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x*w
def loss(xs, ys):
    y_pred = forward(xs)
    loss = (y_pred-ys)**2
    return loss
def gradient(xs, ys):
    return 2*xs*(xs*w-ys)

l = []
l1 = []
for epoch in range(100):
    loss_val = 0
    for x, y in zip(x_data, y_data):
        grad = gradient(x,y)
        w -= 0.01*grad
        loss_val = loss(x,y)#最后一次的loss为准
    l.append(loss_val)
    l1.append(epoch)


def forward(x):
    return x * w

# define the cost function MSE
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# define the gradient function  gd
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val  # 0.01 learning rate
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()


