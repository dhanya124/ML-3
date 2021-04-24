import autograd.numpy as np 
import pandas as pd
from autograd import grad
from scipy.special import expit
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1/(1+np.exp(-1*z))

class LogisticRegression(object):
    """docstring for LogisticRegression"""
    def __init__(self, fit_intercept=True):
        self.fit_intercept=fit_intercept

    def unreguralised_loss(self,theta):
        z=np.dot(self.X_,theta)
        y_pred=sigmoid(z)
        loss=-1*np.mean(self.y*np.log(y_pred)+(1-self.y)*np.log(1-y_pred))
        return loss

    
    
    def L1_regularised_loss(self,theta):
        y_pred=sigmoid(np.dot(self.X_,theta))
        return (-np.sum(self.y*np.log(y_pred)+(1-self.y)*np.log(1-y_pred))+self.lambda1*(np.sum(abs(self.y))))
    
    def L2_regularised_loss(self,theta):
        y_pred=sigmoid(np.dot(self.X_,theta))
        return (-np.sum(self.y*np.log(y_pred)+(1-self.y)*np.log(1-y_pred)))+(self.lambda2*np.linalg.norm(self.theta))
                    
    def fit(self, X, y, n_iter=2000, lr=np.e**-5, lr_type='constant'):
        assert (len(X) == len(y))
        
        # update x based on intercept term
        X_ = X.copy()
        if(self.fit_intercept):
            bias = np.ones((len(X_)))
            X_=np.insert(X_,0,bias,axis=1)

        theta=np.zeros(X_.shape[1])
        n_samples=X_.shape[0]

        # updating the learning rate based on lr_type
        for iter in range(1,n_iter+1):
            if lr_type=='inverse':
                curr_lr=lr/iter
            else:
                curr_lr = lr

            # print(np.dot(X_,theta))
            y_pred=sigmoid(np.dot(X_,theta))

            # updating the coefficients
            # print(y_pred)
            theta-=(curr_lr)*np.dot(X_.T,y_pred-y)

        self.theta=theta
        self.X_=X_

    def fit_L1_regularized(self, X, y,lamda, n_iter=100, lr=0.01, lr_type='constant'):
        # checking whether X and y have same number of samples 
        assert (len(X) == len(y))
        self.lambda1=lamda
        # update x based on intercept term
        X_ = X.copy()
        if(self.fit_intercept):
            bias = np.ones((len(X_)))
            X_=np.insert(X_,0,bias,axis=1)

        theta=np.zeros(X_.shape[1])
        n_samples=X_.shape[0]

        self.X_=X_
        self.y=y

        # updating the learning rate based on lr_type
        for iter in range(1,n_iter+1):
            if lr_type=='inverse':
                curr_lr=lr/iter
            else:
                curr_lr = lr

            # updating the coefficients
            gradient=grad(self.L1_regularised_loss)
            theta=theta-(curr_lr)*(gradient(theta))
        self.theta=theta



    def fit_L2_regularized(self, X, y,lamda, n_iter=10000, lr=0.0001, lr_type='constant'):
            
        # checking whether X and y have same number of samples 
        assert (len(X) == len(y))
        self.lambda2=lamda
        # update x based on intercept term
        self.y=y
        X_=X.copy()
        if(self.fit_intercept):
            bias = np.ones(len(X_))
            X_=np.insert(X_,0,bias,axis=1)

        self.X_=X_
        self.theta=np.ones(self.X_.shape[1])
        self.theta=self.theta/2
        n_samples=self.X_.shape[0]
        
        # updating the learning rate based on lr_type
        for iter in range(1,n_iter+1):
            if lr_type=='inverse':
                curr_lr=lr/iter
            else:
                curr_lr = lr

            # updating the coefficients
            gradient=grad(self.L2_regularised_loss)
            self.theta=self.theta-(curr_lr)*(gradient(self.theta))
    
    def fit_unregularized_autograd(self,X,y,n_iter=4000,lr=(np.e)**-5,lr_type='constant'):
        
        assert (len(X) == len(y))
        # update x based on intercept term
        self.y=y
        X_ = X.copy()
        if(self.fit_intercept):
            bias = np.ones((len(X_)))
            X_=np.insert(X_,0,bias,axis=1)
        
        theta=np.ones(X_.shape[1])
        theta=theta/2
        n_samples=X_.shape[0]
        self.X_=X_
        # updating the learning rate based on lr_type
        for iter in range(1,n_iter+1):
            if lr_type=='inverse':
                curr_lr=lr/iter
            else:
                curr_lr = lr
           
            # updating the coefficients
            gradient=grad(self.unreguralised_loss)
            theta-=(curr_lr) * gradient(theta)
        self.theta=theta


    def predict(self,X_test):
        X=X_test.copy()
        if self.fit_intercept:
            bias=np.ones(len(X))
            X=np.insert(X,0,bias,axis=1)
            
        y_hat=sigmoid(np.dot(X,self.theta))
        for i in range(0,len(y_hat)):
            if y_hat[i]>=0.5:
                y_hat[i]=int(1)
            else:
                y_hat[i]=int(0)                
        return y_hat



        



            
        
        

