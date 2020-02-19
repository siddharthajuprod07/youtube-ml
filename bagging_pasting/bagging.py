import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier

wineData  = pd.read_csv("C:\\Users\\USER\\OneDrive\\youtube_current_content\\bagging and pasting\\winequality-red.csv")
wineData['category'] = wineData['quality'] >= 7
X = wineData[wineData.columns[0:11]].values
y = wineData['category'].values.astype(np.int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=41)
print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)
scaler = StandardScaler()
X_train_stan = scaler.fit_transform(X_train)
logReg = LogisticRegression(solver='lbfgs')
logReg.fit(X_train_stan, y_train)
X_test_stan = scaler.transform(X_test)
y_pred = logReg.predict(X_test_stan)
print('precision on the test set: ', precision_score(y_test, y_pred))
print('accuracy on the test set: ', accuracy_score(y_test, y_pred))
phat = logReg.predict_proba(X_test_stan)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
print('AUC is: ', auc(fpr,tpr))
bagClf = BaggingClassifier(LogisticRegression(random_state=0, solver='lbfgs'), n_estimators = 500, oob_score = True)
bagClf.fit(X_train_stan, y_train)
print(bagClf.oob_score_)
y_pred = bagClf.predict(X_test_stan)
phat = bagClf.predict_proba(X_test_stan)[:,1]
print('precision on the test set: ', precision_score(y_test, y_pred))
print('accuracy on the test set: ', accuracy_score(y_test, y_pred))
fpr, tpr, thresholds = roc_curve(y_test, phat)
plt.subplots(figsize=(8,6))
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
print('AUC is: ', auc(fpr,tpr))

