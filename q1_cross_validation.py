from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold 
from LogisticRegression.logisticRegression import LogisticRegression

def accuracy(y_hat,y):
    n=y.size
    equal=0
    for i in range(n):
        if y_hat[i]==y[i]:
            equal+=1
    return (1.0*equal)/n

(X,y)=load_breast_cancer(return_X_y=True)

k = 3
kf = KFold(n_splits=k, random_state=None)
accuracies=[]
for train_index , test_index in kf.split(X):
    X_train , X_test = X[train_index,:],X[test_index,:]
    y_train , y_test = y[train_index] , y[test_index]
    LR = LogisticRegression(fit_intercept=True)
    LR.fit(X_train, y_train) 
    y_hat = LR.predict(X_test)

    accuracies.append(accuracy(y_test,y_hat))
print(accuracies)
     