#加载数据:乳腺癌检测
from sklearn.datasets.base import load_iris
from sklearn.model_selection._split import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing.data import StandardScaler
import numpy as np
#from . import tools
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')
        
iris=load_iris()
iris_data=iris.data[:, [2, 3]]
print(iris_data)

#切分数据集
X_train, X_test, y_train, y_test=train_test_split(iris_data, iris.target, test_size=0.3, random_state=0)

#
sc = StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

#模型训练与评估，默认L2范数，与model=LogisticRegression(penalty="l2")等价
model=LogisticRegression(C=1000.0, random_state=0)
model.fit(X_train_std, y_train)
model.predict_proba(np.array(X_test_std[0,:]).reshape(1, -1))
plot_decision_regions(X_combined_std, y_combined, classifier=model, test_idx=range(105, 150))

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.xlabel("花瓣长度[标准]")
plt.ylabel("花瓣宽度[标准]")
plt.legend(loc="upper left")
plt.show()