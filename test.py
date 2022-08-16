import numpy as np
import math

class Circle(object):
    pass

circle1=Circle()
circle2=Circle()



# 生成水平视角宽度 a
b = 100**2+9.15**2
a = (100**2+9.15**2)**0.5
c = a**2


#水平视角
angle_ver = math.atan(12.3/a)


a1 = (100**2 + 12.3**2)**0.5
angle_hor = math.atan(9.15/a1)


# 偏转角度
angle_sita = (29.83 + 90)*math.pi/180

#求角度
angle_a = math.atan(9.15/a)

angle_other = math.pi - angle_a - angle_sita
working_distance = 331.72

working_l = math.sin(angle_sita)*working_distance/math.sin(angle_other)
working_y = working_l*math.tan(angle_ver)
working_x = working_l*math.tan(angle_hor)

b=2
# **************************************************求解内参与外参*********************************************************
# import pandas as pd
# # camera_matrix = np.array([[6892.1, 0, 1821.4,0], [0, 4114.8, 1089.8,0],  [0, 0, 1,0]])
# camera_matrix = np.array([[10539.5, 0, 2971.7,0], [0,  8173.26, 1053.24,0],  [0, 0, 1,0]])
# rotation_matrix = np.array([[-0.08175, -0.9965, 0.01947,76.371], [-0.8629,0.06099,-0.5017, 61.683], [0.4987,-0.05781,-0.8648, 593.12],[0,0,0,1]])
#
#
# # A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
# # b = np.transpose(np.array([[-3,5,-2]]))
# # x = np.linalg.solve(A,b)
# # print(x)
#
# #
# all_matrix =np.inner( camera_matrix, rotation_matrix)
#
# a = np.array([1,2,3,4,5])
# print(a)
# print(a[1:5:2])
#
# all_matrix_3 = all_matrix[:, 0:-1]
#
#
# b1 = np.transpose(np.array([[1413, 1078, 1]]))
#
# b2 = np.transpose(np.array([[1702,1085,1]]))
#
# b3 = np.transpose(np.array([[2330.16, 1208.2, 1]]))
# b4 = np.transpose(np.array([[2036.13, 1217.4, 1]]))
# # b = np.array([1413,1078,1])1128.04 951.034
# # 840.389 960.277 2330.16 1208.2
# # 2036.13 1217.4
#
# x1 = np.linalg.solve(all_matrix_3,b1)
#
# x2 = np.linalg.solve(all_matrix_3,b2)
#
#
#
# print(x1)
# print(x2)
# dist = np.linalg.norm(x1 - x2)
# print(dist)
#
# print(26.488/dist)
# x3 = np.linalg.solve(all_matrix_3,b3)
# x4 = np.linalg.solve(all_matrix_3,b4)
# print(x3)
# print(x4)
# dist1 = np.linalg.norm(x3 - x4)
# print(26.488/dist1)



# 乘积
A = np.array([[1,2,3],[2,4,6]]).T
B = np.array([[3,0],[0,1]])
#array生成数组，用np.dot()表示矩阵乘积，（*）号或np.multiply()表示点乘
print(A)
#np.dot() 表示矩阵乘积
print(np.dot(A,B))


#np.mat() 生成矩阵
C = np.mat(A)
D = np.mat(B)
print(np.dot(C,D))

print(C*D)
