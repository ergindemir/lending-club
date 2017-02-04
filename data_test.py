import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from Data import Data

dfloan = pd.read_csv('data/loan.csv',low_memory=False)
data = Data(dfloan)
data.merge_dataset('dummy')
#data.dfmerge.columns
#data.dfmerge.columns.tolist()
X_train, X_test, y_train, y_test = data.get_train_test_set()

data.exclude_columns=['grade']
data.reset()
data.merge_dataset('dummy')
data.dfmerge.columns.tolist()
X_train, X_test, y_train, y_test = data.get_train_test_set()

data.merge_dataset(type='dummy',exclude_columns=['sub_grade'])
sorted(data.dfmerge.columns.tolist())
X_train, X_test, y_train, y_test = data.get_train_test_set()
model = RandomForestClassifier(n_estimators=600, n_jobs = -1)
model.fit(X_train,y_train)
model.score(X_test,y_test)
data.dfmerge.columns[np.argsort(model.feature_importances_)[::-1]]


data.merge_dataset(type='dummy',exclude_columns=['sub_grade'])
sorted(data.dfmerge.columns.tolist())
X_train, X_test, y_train, y_test = data.get_train_test_set()
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
scores = model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, scores)
plt.plot(fpr,tpr)


data.merge_dataset(exclude_columns=['sub_grade','Volatility'])
sorted(data.dfmerge.columns.tolist())
X_train, X_test, y_train, y_test = data.get_train_test_set()
model = RandomForestClassifier(n_estimators=400, n_jobs = -1)
model.fit(X_train,y_train)
model.score(X_test,y_test)
data.dfmerge.columns[np.argsort(model.feature_importances_)[::-1]]
score = (model.predict(X_test)==y_test).astype(int).to_frame('score')
df = pd.concat([score, data.df], axis=1, join='inner')
df.groupby(df.Date.dt.year).score.mean().plot()

data.merge_dataset(exclude_columns=['sub_grade'])
sorted(data.dfmerge.columns.tolist())
X_train, X_test, y_train, y_test = data.get_train_test_set()
model = RandomForestClassifier(n_estimators=400, n_jobs = -1)
model.fit(X_train,y_train)
model.score(X_test,y_test)
data.dfmerge.columns[np.argsort(model.feature_importances_)[::-1]]
score = (model.predict(X_test)==y_test).astype(int).to_frame('score')
df = pd.concat([score, data.df], axis=1, join='inner')
df.groupby(df.Date.dt.year).score.mean().plot()

