import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X,Y=load_digits(return_X_y=True)
pca=PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

x=X[:,0]
y=X[:,1]

fig,ax=plt.subplots()
for digit in np.unique(Y):
    i=np.where(Y==digit)
    ax.scatter(x[i],y[i],label=digit)

ax.legend()

fig.savefig('q3_fig.png')