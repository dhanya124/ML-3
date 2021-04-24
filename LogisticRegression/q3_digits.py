import numpy as np 
import pandas as pd 
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_digits
from sklearn import preprocessing
# from mc_dummy import Multiclass
from Multiclass import MulticlassLR
from sklearn import preprocessing


def accuracy(y_hat,y):
    n=y.size
    equal=0
    for i in range(n):
        if y_hat[i]==y[i]:
            equal+=1
    return (1.0*equal)/n


(data,labels)=load_digits(return_X_y=True)

n_class=len(np.unique(labels))
Obj=MulticlassLR(n_class)
Obj.fit(data,labels)

y_hat=Obj.predict(data)
print(y_hat)
print(labels)
print("--------------------------------")
print(accuracy(y_hat,labels))





'''
autograd: 1000 iter and 0.1 learning rate
gradient_descent: 1000 iter and 0.001 learning rate

'''
