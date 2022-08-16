import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('https://blog.caiyongji.com/assets/mouse_viral_study.csv')
df.head()
print(df.head())
sns.scatterplot(x='Med_1_mL',y='Med_2_mL',hue='Virus Present',data=df)
sns.pairplot(df,hue='Virus Present')


# SVC: Supprt Vector Classifier支持向量分类器
from sklearn.svm import SVC

# 准备数据
y = df['Virus Present']
X = df.drop('Virus Present', axis=1)



#遍历所有的核函数，找到最佳的超参数
# from sklearn.model_selection import GridSearchCV
# svm = SVC()
# param_grid = {'C':[0.01,0.1,1],'kernel':['rbf','poly','linear','sigmoid'],'gamma':[0.01,0.1,1]}
# grid = GridSearchCV(svm,param_grid)
# grid.fit(X,y)
# print("grid.best_params_ = ",grid.best_params_,", grid.best_score_ =" ,grid.best_score_)


# 定义模型
model = SVC(kernel='rbf', C=1,gamma=0.01)

# 训练模型
model.fit(X, y)


# 绘制图像
# 定义绘制SVM边界方法
def plot_svm_boundary(model, X, y):
    X = X.values
    y = y.values

    # Scatter Plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='coolwarm')

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


plot_svm_boundary(model, X, y)