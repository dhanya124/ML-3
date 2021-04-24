import autograd.numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from autograd import grad

onehotencoder=OneHotEncoder(sparse=False)

def softmax(y):
    e_y = (np.e) ** (y - np.max(y))
    y=(e_y) / (e_y.sum()).reshape(-1,1)
    return y 

class MulticlassLR():

    def __init__(self,n_class,fit_intercept=True,opt='grad_desc'):        
        self.fit_intercept=fit_intercept
        self.n_class=n_class 
        self.opt=opt
        
    def loss(self,weights):
        """
        Y: onehot encoded
        """
        z = - self.X_ @ weights
        loss= 1/(self.n_samples) * (np.trace(self.X_ @ weights @ self.y_encoded.T) + np.sum(np.log(np.sum(np.exp(z), axis=1))))
        return loss



    def fit(self,X,y,n_iter=1000,lr=0.001):
        X_=X.copy()
        if self.fit_intercept:
            bias=np.ones(len(X_))
            X_=np.insert(X_,0,bias,axis=1)
        
        self.X_=X_
        self.y_encoded=onehotencoder.fit_transform(y.reshape(-1,1))
        self.n_samples=len(X_)

        # initializing the weights
        weights=np.zeros((X_.shape[1],self.y_encoded.shape[1]))

        for i in range(n_iter):
            if self.opt=='autograd':
                gradient=grad(self.loss)
                weights-=((lr)*(gradient(weights)))
            elif self.opt=='grad_desc':
                z=-np.dot(X_,weights)
                p=softmax(z)
                weights-=(lr/self.n_samples)*(X_.T@(self.y_encoded-p))

        self.weights=weights



    def predict(self,X):
        X_=X.copy()
        if self.fit_intercept:
            bias=np.ones(len(X_))
            X_=np.insert(X_,0,bias,axis=1)
        z=-np.dot(X_,self.weights)
        p=softmax(z)
        return np.argmax(p,axis=1)
            
        






            