import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.model_selection import KFold 
from LogisticRegression.logisticRegression import LogisticRegression

(data,labels)=load_breast_cancer(return_X_y=True)
scaler = preprocessing.MinMaxScaler()
data = scaler.fit_transform(data)

k = 3
kf_out= KFold(n_splits=k, random_state=None,shuffle=True)
kf_in = KFold(n_splits=k, random_state=None,shuffle=True)
fit_acc=[]

lamda1=np.linspace(0,1,1000)
lamda2=np.linspace(0,1,1000)

print("L1-------")
accuracy_final=0
for train_idx , test_idx in kf_out.split(X):
    X_train , X_test = data[train_idx,:],data[test_idx,:]
    y_train , y_test = labels[train_idx] , labels[test_idx]
    best=0
    validation=[]
    for l in lamda1:
        acc=0
        for train_idx_,test_idx_ in kf_in.split(X_train):
            X_train_in ,X_val= X_train[train_idx_],X_train[test_idx_]
            y_train_in,y_val = y_train[train_idx_],y_train[test_idx_]
            model = LogisticRegression(fit_intercept=True)
            model.fit_L1_regularized(X_train_in, y_train_in,lamda=l) 
            y_hat = model.predict(X_val)
            acc += accuracy((y_val),(y_hat))
            acc=acc/(3.0)
        if(acc>best):
            best = acc
            best_parameter= lamda
        validation.append(acc)
    model.fit_L1_regularized(X_train, y_train,lamda=best_parameter) 
    y_hat = model.predict(X_test)
    accuracy_final += accuracy((y_hat),(y_test))

print(accuracy_final)
print(best_parameter)


print("L2-------------")
accuracy_final=0
for train_idx , test_idx in kf_out.split(X):
    X_train , X_test = data[train_idx,:],data[test_idx,:]
    y_train , y_test = labels[train_idx] , labels[test_idx]
    best=0
    validation=[]
    for l in lamda2:
        acc=0
        for train_idx_,test_idx_ in kf_in.split(X_train):
            X_train_in ,X_val= X_train[train_idx_],X_train[test_idx_]
            y_train_in,y_val = y_train[train_idx_],y_train[test_idx_]
            model = LogisticRegression(fit_intercept=True)
            model.fit_L2_regularized(X_train_in, y_train_in,lamda=l) 
            y_hat = model.predict(X_val)
            acc += accuracy((y_val),(y_hat))
            acc=acc/(3.0)
        if(acc>best):
            best = acc
            best_parameter= lamda
        validation.append(acc)
    model.fit_L2_regularized(X_train, y_train,lamda=best_parameter) 
    y_hat = model.predict(X_test)
    accuracy_final += accuracy((y_hat),(y_test))

print(accuracy_final)
print(best_parameter)
