import numpy as np 
import pandas as pd 
from sklearn.datasets import load_digits 
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
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


k = 3
kf = KFold(n_splits=k, random_state=None)
errors=[]
for train_index , test_index in kf.split(data):
    X_train , X_test = data[train_index,:],data[test_index,:]
    y_train , y_test = labels[train_index] , labels[test_index]
    nn1 = fcmlp(10,0.01,data,labels,[1],['relu'])
    nn1.fit() 
    y_hat = nn1.predict(X_test)

    errors.append(rmse(y_hat,y_test))

print(errors)

