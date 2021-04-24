import numpy as np 
import pandas as pd 
from sklearn.datasets import load_digits 
from sklearn import preprocessing
from NN import fcmlp

def accuracy(y_hat,y):
    n=y.size
    equal=0
    for i in range(n):
        if y_hat[i]==y[i]:
            equal+=1
    return (1.0*equal)/n

def rmse(y_hat,y):
    return np.sqrt(np.mean(np.square(y_hat-y)))


(data,labels)=load_digits(return_X_y=True)
scaler=preprocessing.MinMaxScaler()
data=scaler.fit_transform(data)

labels=np.matrix(labels).T


print(labels.shape)
print("----------")
nn1=fcmlp(10,0.01,data,labels,[1],['relu'])
nn1.fit()

y_hat=nn1.predict()
print(y_hat)
# print(nn1.weights)
# print(labels)

print(rmse(y_hat,labels))