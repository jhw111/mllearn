'''
Created on 2019年10月23日

@author: havery
'''
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
X_breast, y_breast = load_breast_cancer(return_X_y=True)
X_breast_train, X_breast_test, y_breast_train, y_breast_test = train_test_split(X_breast, y_breast, stratify= y_breast, random_state=0, test_size=0.3)


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(MinMaxScaler(), LogisticRegression(solver='saga', multi_class='auto', random_state=42, max_iter=5000))

#优化LogisticRegression分类器的C和penalty参数
from sklearn.model_selection import GridSearchCV
param_grid = {'logisticregression__C':[0.1, 1.0, 10], 'logisticregression__penalty':['l1','l2']}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=1, return_train_score=True)
grid.fit(X_breast, y_breast)

'''
在拟合网格搜索对象时，会在训练集上找到最佳的参数组合（使用交叉验证）。
 可以通过访问属性cv_results_来得到网格搜索的结果。 通过这个属性允许可以检查参数对模型性能的影响
'''
import pandas as pd
from matplotlib import pyplot as plt
df_grid = pd.DataFrame(grid.cv_results_)
print(df_grid)
#最佳参数
print(grid.best_params_)

#df_grid[['mean_train_score', 'std_test_score']].boxplot()
#plt.show()

#梯度下降法
#评估hinge(铰链) 和log(对数)损失之间的差异,微调penalty
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
pipe = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000))
param_grid ={'sgdclassifier__loss': ['hinge', 'log'],'sgdclassifier__penalty': ['l2', 'l1']}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=1)
scores = cross_validate(pipe, X_breast, y_breast, scoring='balanced_accuracy', cv=3, return_train_score=True)
df_scores = pd.DataFrame(scores)
df_scores[['train_score', 'test_score']].boxplot()
grid.fit(X_breast_train, y_breast_train)
plt.show()

