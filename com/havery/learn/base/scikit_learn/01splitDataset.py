'''
Created on 2019年10月23日

@author: havery
'''
#train_test_split 是一个用于将数据拆分为两个独立数据集的效用函数
from sklearn.model_selection import train_test_split
from sklearn.datasets.base import load_iris
#stratify参数可强制将训练和测试数据集的类分布与整个数据集的类分布相同
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify = iris.target, random_state=42)

#逻辑回归分类器
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=5000, random_state=42)
#使用fit方法学习机器学习模型
clf.fit(X_train, y_train)
#使用score方法来测试此方法，依赖于默认的准确度指标
accuracy=clf.score(X_test, y_test)
print('{}逻辑回归的精确度得分是：{}'.format(clf.__class__.__name__, accuracy))

#随机森林分类器
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
clf2.fit(X_train, y_train)
accuracy2 = clf2.score(X_test, y_test)
print('{}随机森林的精确度得分是：{}'.format(clf2.__class__.__name__, accuracy2))

#梯度提升决策树分类器
from sklearn.datasets import load_breast_cancer
X_breast, y_breast = load_breast_cancer(return_X_y=True)
X_breast_train, X_breast_test, y_breast_train, y_breast_test = train_test_split(X_breast, y_breast, stratify= y_breast, random_state=0, test_size=0.3)
from sklearn.ensemble import GradientBoostingClassifier
clf3 = GradientBoostingClassifier(n_estimators=100, random_state=0)
clf3.fit(X_breast_train, y_breast_train)
#使用拟合分类器预测测试集的分类标签
y_pred=clf3.predict(X_breast_test)
#计算测试集的balanced_accuracy_score精度
from sklearn.metrics import balanced_accuracy_score
accuracy3 = balanced_accuracy_score(y_breast_test, y_pred)
print('{}梯度提升决策树的精确度得分是：{}'.format(clf3.__class__.__name__, accuracy3))