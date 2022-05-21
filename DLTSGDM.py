import numpy as np
import pandas as pd
import math

# l=[0,-7.954576953947253,
#  -1.831736952648043,
#  0.9564038732394238,
#  -1206.012352833175,
#  -0.002211408914881474,
#  1.007397296772297,
#  7.960648549276812,
#  -9931.930426741019,
#  0.0001906592538635787,
#  0.001951427147760754,
#  -0.0007626684312629084]

l=[0,3.125470808323541,
 -0.004801647271067688,
 1.110835966793317,
 -2832.142296195932,
 -0.04681661061834985,
 3.318909725935214,
 0.1398269294385805,
 93.4141468525022,
 -0.0002231458380250097,
 -2.569403815866898e-05,
 0.0006125970254814428]

l = np.array(l)


pd_data =pd.read_csv('gcpPoints3.csv')
data = np.array(pd_data.iloc[0:,1:])

# X = data[:,:1]
# Y = data[:,1:2]
# Z = data[:,2:3]
# U = data[:,3:4]
# V = data[:,4:5]

X = data[:,:1]
Y = data[:,1:2]
Z = data[:,2:3]
U = data[:,3:4]
V = data[:,4:5]
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

#X, X_test, y_train, y_test = train_test_split(X, y,test_size=3, random_state=0)
# u = (l[1]*x+l[2]*y+l[3]*z +l[4])/(l[9]*x+l[10]*y+l[11]*z+1)
# v = (l[5]*x+l[6]*y+l[7]*z +l[8])/(l[9]*x+l[10]*y+l[11]*z+1)

def forward(x,y,z):
   u = -(l[1] * x + l[2] * y + l[3] * z + l[4]) / (l[9] * x + l[10] * y + l[11] * z + 1)
   v = -(l[5] * x + l[6] * y + l[7] * z + l[8]) / (l[9] * x + l[10] * y + l[11] * z + 1)
   return u,v


def cost(X,Y,Z,U,V):
    cost =0
    for x,y,z,u,v in zip(X,Y,Z,U,V):
        u_,v_ = forward(x,y,z)
        cost +=(u_ - u) **2+ (v_-v)**2
    return cost/len(X)


def gradient(X,Y,Z,U,V):
   #u_, v_ = forward(x, y, z)
   m = np.arange(1, 13, 1)
   for x,y,z,u,v in zip(X,Y,Z,U,V):
       mu = -(l[1] * x + l[2] * y + l[3] * z + l[4]) / (l[9] * x + l[10] * y + l[11] * z + 1)
       nu = -(l[5] * x + l[6] * y + l[7] * z + l[8]) / (l[9] * x + l[10] * y + l[11] * z + 1)
       p1 =  l[1] * x + l[2] * y + l[3] * z + l[4]
       p2 = l[5] * x + l[6] * y + l[7] * z + l[8]
       p = (l[9] * x + l[10] * y + l[11] * z + 1)
       m[1] =- 2 * (mu - u) * x / p
       m[2] =- 2 * (mu - u) * y / p
       m[3] = -2 * (mu - u) * z / p
       m[4] =- 2 * (mu - u) * 1 / p
       m[5] = -2 * (nu - v) * x / p
       m[6] = -2 * (nu - v) * y / p
       m[7] =- 2 * (nu - v) * z / p
       m[8] = -2 * (nu - v) * 1 / p

       # m[9] = -2 * (mu - u) *  (l[1] * x + l[2] * y + l[3] * z + l[4])* x / p**2 -\
       #        2 * (nu - v) *(l[5] * x + l[6] * y + l[7] * z + l[8])* x / p**2
       # m[10] = -2 * (mu - u) * (l[1] * x + l[2] * y + l[3] * z + l[4]) * y /p ** 2 - \
       #        2 * (nu - v) * (l[5] * x + l[6] * y + l[7] * z + l[8]) * y / p ** 2
       # m[11] = -2 * (mu - u) * (l[1] * x + l[2] * y + l[3] * z + l[4]) * z /p ** 2 - \
       #        2 * (nu - v) * (l[5] * x + l[6] * y + l[7] * z + l[8]) *  z/p ** 2
       m[9]=0
       m[10]=0
       m[11]=0
   return m/len(x)

# loss_val = 0
# for x,y,z,u,v in zip(X, Y, Z, U, V):
#    loss_val += loss(x,y,z,u,v)
# loss_all =math.sqrt(loss_val/X.size)
# print(l)
# print(m)
for epoch in range(500):
  w = 0.000000001*gradient(X, Y, Z, U, V)
  loss_val = cost(X, Y, Z, U, V)
  l -= w
loss_val = 0
for x,y,z,u,v in zip(X, Y, Z, U, V):
   loss_val += cost(x,y,z,u,v)
loss_val = math.sqrt(loss_val / X.size)
print(l)
print("epoch",epoch,"loss_val", loss_val)

X1 = data[0:83,:1]
Y1 = data[0:83,1:2]
Z1 = data[0:83,2:3]
U1 = data[0:83,3:4]
V1 = data[0:83,4:5]

loss_ =0
for x,y,z,u,v in zip(X1, Y1, Z1, U1, V1):
    loss_+=cost(x,y,z,u,v)
loss_ = math.sqrt(loss_ / X1.size)
print(loss_)


# def gradient(x,y,z,u,v):
#    #u_, v_ = forward(x, y, z)
#    mu = -(l[1] * x + l[2] * y + l[3] * z + l[4]) / (l[9] * x + l[10] * y + l[11] * z + 1)
#    nu = -(l[5] * x + l[6] * y + l[7] * z + l[8]) / (l[9] * x + l[10] * y + l[11] * z + 1)
#    p = (l[9] * x + l[10] * y + l[11] * z + 1)
#    m[1] =- 2 * (-(l[1] * x + l[2] * y + l[3] * z + l[4]) / (l[9] * x + l[10] * y + l[11] * z + 1) - u) * x / (l[9] * x + l[10] * y + l[11] * z + 1)
#    m[2] =- 2 * (-(l[1] * x + l[2] * y + l[3] * z + l[4]) / (l[9] * x + l[10] * y + l[11] * z + 1) - u) * y / (l[9] * x + l[10] * y + l[11] * z + 1)
#    m[3] = -2 * (-(l[1] * x + l[2] * y + l[3] * z + l[4]) / (l[9] * x + l[10] * y + l[11] * z + 1) - u) * z / (l[9] * x + l[10] * y + l[11] * z + 1)
#    m[4] =- 2 * (-(l[1] * x + l[2] * y + l[3] * z + l[4]) / (l[9] * x + l[10] * y + l[11] * z + 1) - u) * 1 / (l[9] * x + l[10] * y + l[11] * z + 1)
#    m[5] = -2 * (-(l[5] * x + l[6] * y + l[7] * z + l[8]) / (l[9] * x + l[10] * y + l[11] * z + 1) - v) * x / (l[9] * x + l[10] * y + l[11] * z + 1)
#    m[6] = -2 * (-(l[5] * x + l[6] * y + l[7] * z + l[8]) / (l[9] * x + l[10] * y + l[11] * z + 1) - v) * y / (l[9] * x + l[10] * y + l[11] * z + 1)
#    m[7] =- 2 * (-(l[5] * x + l[6] * y + l[7] * z + l[8]) / (l[9] * x + l[10] * y + l[11] * z + 1) - v) * z / (l[9] * x + l[10] * y + l[11] * z + 1)
#    m[8] = -2 * (-(l[5] * x + l[6] * y + l[7] * z + l[8]) / (l[9] * x + l[10] * y + l[11] * z + 1) - v) * 1 / (l[9] * x + l[10] * y + l[11] * z + 1)
#
#    m[9] = 2 * ((l[1] * x + l[2] * y + l[3] * z + l[4]) / (l[9] * x + l[10] * y + l[11] * z + 1) - u) *  (l[1] * x + l[2] * y + l[3] * z + l[4])* x / (l[9] * x + l[10] * y + l[11] * z + 1)**2 +\
#           2 * ((l[5] * x + l[6] * y + l[7] * z + l[8]) / (l[9] * x + l[10] * y + l[11] * z + 1) - v) *(l[5] * x + l[6] * y + l[7] * z + l[8])* x / (l[9] * x + l[10] * y + l[11] * z + 1)**2
#    m[10] = 2 * ((l[1] * x + l[2] * y + l[3] * z + l[4]) / (l[9] * x + l[10] * y + l[11] * z + 1) - u) * (l[1] * x + l[2] * y + l[3] * z + l[4]) * y / (l[9] * x + l[10] * y + l[11] * z + 1) ** 2 + \
#           2 * ((l[5] * x + l[6] * y + l[7] * z + l[8]) / (l[9] * x + l[10] * y + l[11] * z + 1) - v) * (l[5] * x + l[6] * y + l[7] * z + l[8]) * y / (l[9] * x + l[10] * y + l[11] * z + 1) ** 2
#    m[11] = 2 * ((l[1] * x + l[2] * y + l[3] * z + l[4]) / (l[9] * x + l[10] * y + l[11] * z + 1) - u) * (l[1] * x + l[2] * y + l[3] * z + l[4]) * z / (l[9] * x + l[10] * y + l[11] * z + 1) ** 2 + \
#           2 * ((l[5] * x + l[6] * y + l[7] * z + l[8]) / (l[9] * x + l[10] * y + l[11] * z + 1) - v) * (l[5] * x + l[6] * y + l[7] * z + l[8]) *  z/ (l[9] * x + l[10] * y + l[11] * z + 1) ** 2
#
#    return m