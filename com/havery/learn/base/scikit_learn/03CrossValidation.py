'''
Created on 2019年10月23日

@author: havery
'''
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
X_breast, y_breast = load_breast_cancer(return_X_y=True)
X_breast_train, X_breast_test, y_breast_train, y_breast_test = train_test_split(X_breast, y_breast, stratify= y_breast, random_state=0, test_size=0.3)
'''
分割数据减少了可用于学习模型的样本数量,应尽可能使用交叉验证
有多个拆分也会提供有关模型稳定性的信息
'''
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(MinMaxScaler(), LogisticRegression(solver='lbfgs', multi_class='auto', random_state=42))
scores = cross_validate(pipe, X_breast_train, y_breast_train, cv=7, return_train_score = True)

#使用pandas、matplotlib快速绘图(箱图)
import pandas as pd
from matplotlib import pyplot as plt
#df_scores = pd.DataFrame(scores)
#df_scores[['train_score', 'test_score']].boxplot()
#plt.show()

#梯度下降法
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
pipe = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000))
scores = cross_validate(pipe, X_breast, y_breast, scoring='balanced_accuracy', cv=3, return_train_score=True)
df_scores = pd.DataFrame(scores)
df_scores[['train_score', 'test_score']].boxplot()
plt.show()