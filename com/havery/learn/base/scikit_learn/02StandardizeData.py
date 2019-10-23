'''
Created on 2019年10月23日

@author: havery
'''
#train_test_split 是一个用于将数据拆分为两个独立数据集的效用函数
from sklearn.model_selection import train_test_split
from sklearn.datasets.base import load_iris
from sklearn.metrics import balanced_accuracy_score
#stratify参数可强制将训练和测试数据集的类分布与整个数据集的类分布相同
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, stratify = iris.target, random_state=42)


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
'''
MinMaxScaler变换器用于规范化数据
学习（即，fit方法）训练集上的统计数据并标准化（即，transform方法）训练集和测试集。 最后训练和测试这个模型并得到归一化后的数据集。
'''
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)
accuracy=clf.score(X_test_scaled, y_test)
print('{}逻辑回归的精确度得分是：{}'.format(clf.__class__.__name__, accuracy))
print('{}需要{}迭代'.format(clf.__class__.__name__, clf.n_iter_[0]))

'''
错误的预处理模式
1.在整个数据集分成训练和测试集之前标准化数据
X_scaled = scaler.fit_transform(iris)
2.独立地标准化训练和测试集，训练和测试集的标准化不同。
X_train_prescaled = scaler.fit_transform(X_train)
X_test_prescaled = scaler.fit_transform(X_test)
使用管道防止这种错误
'''
from sklearn.pipeline import Pipeline
pipe = Pipeline(steps=[('scaler', MinMaxScaler()),('clf', LogisticRegression(solver='lbfgs', multi_class='auto', random_state=42))])
#或者使用自动命名
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(MinMaxScaler(), LogisticRegression(solver='lbfgs', multi_class='auto', random_state=42))
'''
使用fit来训练分类器和socre来检查准确性
调用fit会调用管道中所有变换器的fit_transform方法。
 调用score（或predict和predict_proba）将调用管道中所有转换器的内部变换。
'''
pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)
print('{}逻辑回归的精确度得分是：{}'.format(pipe.__class__.__name__, accuracy))
print('{}需要{}迭代'.format(pipe.__class__.__name__, clf.n_iter_[0]))
#使用get_params()检查管道的所有参数
#print(pipe.get_params())


from sklearn.datasets import load_breast_cancer
X_breast, y_breast = load_breast_cancer(return_X_y=True)
X_breast_train, X_breast_test, y_breast_train, y_breast_test = train_test_split(X_breast, y_breast, stratify= y_breast, random_state=0, test_size=0.3)
#梯度下降法
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000))
pipe.fit(X_breast_train, y_breast_train)
y_pred = pipe.predict(X_breast_test)
accuracy = balanced_accuracy_score(y_breast_test, y_pred)
print('{}梯度下降的精确度得分是：{}'.format(pipe.__class__.__name__, accuracy))
print('{}需要{}迭代'.format(pipe.__class__.__name__, clf.n_iter_[0]))