from LogisticRegression.logisticRegression import LogisticRegression
import numpy as np 
import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing


def accuracy(y_hat,y):
    n=y.size
    equal=0
    for i in range(n):
        if y_hat[i]==y[i]:
            equal+=1
    return (1.0*equal)/n

(data,target)=load_breast_cancer(return_X_y=True)
scaler=preprocessing.MinMaxScaler()
data=scaler.fit_transform(data)


Lr=LogisticRegression(fit_intercept=True)
Lr.fit_L2_regularized(X=data,y=target,lamda=0.01)
# # print(target)
# # # print("--------------------------------------------------")
# # print(Lr.theta)
y_hat=Lr.predict()
print("--------------------------------------------------")
print(y_hat)
print("--------------------------------------------------")
print("accuracy of my model",accuracy(y_hat,target))

