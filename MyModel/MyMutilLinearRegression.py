#多项式线性回归
from sklearn.datasets import load_boston
from sklearn.model_selection._split import train_test_split
from sklearn.linear_model.base import LinearRegression
from sklearn.preprocessing.data import PolynomialFeatures
from keras import metrics
from sklearn.model_selection._validation import cross_val_predict
#boston房价数据集
boston = load_boston()
#print(boston)
#通过DESCR属性可以查看数据集的详细情况，这里数据有14列，前13列为特征数据，最后一列为标签数据。
#print(boston.DESCR)
#boston的data和target分别存储了特征和标签
#print(boston.data)
#print(boston.target)
#切分数据集
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=2) 


#增加特征多项式让线性回归模型更好地拟合数据
#多项式的个数的不断增加，可以在训练集上有很好的效果，但很容易造成过拟合
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly =poly.fit_transform(X_train)
X_test_poly =poly.fit_transform(X_test)
#多项式线性回归
model2 = LinearRegression(normalize=True)
model2.fit(X_train_poly, y_train)
mutilScore = model2.score(X_test_poly, y_test)
print(mutilScore)

#模型测试，并利用均方根误差(MSE)对测试结果进行评价
#模型的拟合值
y_pred=model2.predict(X_test_poly)
print("MSE:",metrics.mean_squared_error(y_test, y_pred))

#交叉验证
predicted=cross_val_predict(model2, boston.data, boston.target, cv=10)
print ("MSE:", metrics.mean_squared_error(boston.target, predicted))

#画图
import matplotlib.pyplot as plt
plt.scatter(boston.target, predicted, color="y", marker="o")
plt.scatter(boston.target, boston.target, color="g", marker="+")
plt.show()
