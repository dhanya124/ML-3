import autograd.numpy as anp 
import numpy as np
import pandas as pd 
from autograd import grad

def sigmoid(z):
    return 1/(1+(np.e)**(-z))

def identity(z):
    return z

def relu(z):
    return np.maximum(0,z)


class fcmlp:
    def __init__(self,n_iter,lr,X,y,n_nuerons,actfns):
        self.n_iter=n_iter
        self.lr=lr
        self.X=X
        self.y=y
        self.n_nuerons=n_nuerons
        self.n_layers=len(n_nuerons)
        self.actfns=actfns
        self.n_samples=X.shape[0]
        self.n_infeatures=X.shape[1]
    
    def init_weights(self):
        weights={}
        n_prevlayer=self.X.shape[1]
        for i in range(len(self.n_nuerons)):
            a=np.ones((self.n_nuerons[i],n_prevlayer+1)) 
            n_prevlayer=self.n_nuerons[i]
            weights[i+1]=np.array(a)
        l=len(self.n_nuerons)
        weights[len(self.n_nuerons)+1]=np.array(np.ones((1,self.n_nuerons[l-1]+1)))
        return weights

    def com_err(self,weights):
        # This function does one and forward pass and one backward pass and returns the mse error
        prev_acts=(self.X)

        for i in range(len(weights)-1):  
            bias=np.ones(len(prev_acts))
            prev_acts=np.insert(prev_acts,0,bias,axis=1)
            z=np.dot(prev_acts,weights[i+1].T)    
            str=self.actfns[i]
            if str=='relu':
                a=relu(z)
            elif str=='identity':
                a=identity(z)
            elif str=='sigmoid':
                a=sigmoid(z)            
            prev_acts=a
                    
        bias=np.ones(len(prev_acts))
        prev_acts=np.insert(prev_acts,0,bias,axis=1)

        y_pred=np.dot(prev_acts,weights[len(weights)].T)
        y_pred=np.array(sigmoid(y_pred))

        return ((np.square(np.subtract(self.y,y_pred))).mean())



    def fit(self):
        weights=self.init_weights()
        for i in range(self.n_iter):
            grad_=grad(self.com_err)
            c=grad_(weights)

            for j in range(len(weights)):
                weights[j+1]-=(self.lr)*(c[j+1])

            print("iter ",i)
        self.weights=weights
                    
    def predict(self,X):

        prev_layeract=X
        for i in range(len(self.weights)-1):  
            bias=np.ones(len(prev_layeract))
            prev_layeract=np.insert(prev_layeract,0,bias,axis=1)
            z=np.dot(prev_layeract,self.weights[i+1].T)    
            str=self.actfns[i]
            if str=='relu':
                a=relu(z)
            elif str=='identity':
                a=identity(z)
            elif str=='sigmoid':
                a=sigmoid(z)            
            prev_layeract=a
                    
        bias=np.ones(len(prev_layeract))
        prev_layeract=np.insert(prev_layeract,0,bias,axis=1)

        y_hat=np.dot(prev_layeract,self.weights[len(self.weights)].T)
        # y_hat=sigmoid(y_hat)
        return y_hat 
    



