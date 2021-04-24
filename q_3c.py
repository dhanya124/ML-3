import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from Multiclass import MulticlassLR

skf = StratifiedKFold(n_splits=4,random_state=None, shuffle=True)

(data,labels)= load_digits(return_X_y=True)  
accuracies=[]

for train_idx, test_idx in skf.split(data, labels):
    X_train, X_test = data[train_idx], data[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    mclr = MulticlassLR(fit_intercept=True)
    mclr.fit(X_train, y_train,n_iter=60) 
    y_hat = mclr.predict(X_test)
    acc= accuracy(y_hat,y_test)
    accuracies.append(acc)
    print(confusion_matrix(y_test, y_hat))

print(accuracies)
print(np.sum(accuracies)/len(accuracies))


svm=(sns.heatmap(confusion_matrix(y_test, y_hat), annot=True))
figure = svm.get_figure()    
figure.savefig('svm_conf.png', dpi=200)

# Accuracy for 5 folds: [0.8644444444444445, 0.8930957683741648, 0.8975501113585747, 0.8596881959910914 ,0.8786946300420688]
