#加载数据:乳腺癌检测
from sklearn.datasets.base import load_breast_cancer
from sklearn.model_selection._split import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
cancer=load_breast_cancer()
print(cancer)

#切分数据集
X_train, X_test, y_train, y_test=train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=33)

#模型训练与评估，默认L2范数，与model=LogisticRegression(penalty="l2")等价
model=LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print("score:", score)

#L1范数训练与评估
model1 = LogisticRegression(penalty="l1")
model1.fit(X_train, y_train)

score1 = model1.score(X_test, y_test)
print("score1:", score1)

y_predict = model1.predict(X_test)
print(y_predict)

#画图
import matplotlib.pyplot as plt
plt.scatter(cancer.target, y_predict, color="y", marker="o")
plt.scatter(cancer.target, cancer.target, color="g", marker="+")
plt.show()