import numpy as np
import matplotlib as plt

def loaddata(filename):
    file = open(filename)
    x = []
    y = []
    for line in file.readlines():
        line = line.strip().split()
        x.append([1,float(line[0]),float(line[1])])
        y.append([float(line[-1])])
    xmat = np.mat(x)
    ymat = np.mat(y)
    file.close()
    return xmat,ymat

xmat,ymat = loaddata('data.txt')
print("xmat=",xmat,xmat.shape)
print("ymat=",ymat,ymat.shape)

def w_calc(xmat,ymat,alpha=0.01,maxIter=1000):
    w = np.mat(np.random.randn(3,1))
    for i in range(maxIter):
        H =1/(1+np.exp(-xmat*w))
        dw = xmat.T * (H-ymat)
        w -= alpha*dw
    return w

W = w_calc(xmat,ymat)
print("w=",W)
# plt.scatter(xmat[:,1][ymat==0].A,xmat[:2][ymat==0].A)
# plt.show()
# a= 5