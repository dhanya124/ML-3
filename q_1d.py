from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
import matplotlib.pyplot as plt
import mlxtend
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
from LogisticRegression.logisticRegression import LogisticRegression

(X,y)= load_breast_cancer(return_X_y=True)   
scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)

X=X[:,:2]
 
# Initializing Classifiers
model = LogisticRegression(fit_intercept=True)

 
# %matplotlib inline  
 
gs = gridspec.GridSpec(3, 2)
 
fig = plt.figure(figsize=(14,10))
 
labels = ['Logistic Regression']
for model, lab, grd in zip([model],
                         labels,
                         [(0,0)]):
 
    model.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, model=model, legend=2)
    plt.title(lab)
 
figure = fig.get_figure()    
figure.savefig('q_1d.png', dpi=200)

