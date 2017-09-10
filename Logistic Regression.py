import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gendata():
    l1 = [1]*10000
    A1 = np.random.normal(2, 0.5, 10000)
    A2 = np.random.normal(2, 0.5, 10000)
    A = np.column_stack((A1,A2))
    l0 = [0]*10000
    B1 = np.random.normal(0, 0.5, 10000)
    B2 = np.random.normal(0, 0.5, 10000)
    B = np.column_stack((B1,B2))
    X = np.vstack((A,B))
    Y = np.vstack((l1,l0))
    return X, Y

def cost(Y_hat,Y):
    """
    Y_hat: N x 1
    Y: N x 1
    """
    Y_hat =Y_hat.flatten()
    Y =Y.flatten()
    cost1 = 0
    elp = 0.0000000000000000000000000000000000000000000001
    for i in range(len(Y)):
        cost1 -= Y[i]*np.log(Y_hat[i]+elp) + (1-Y[i])*np.log(1-Y_hat[i]+elp)
    return cost1

def error_rate(P,Y):
    return np.mean(Y != P)

class logistic_regression(object):
    def fit(self, X, Y, learning_rate=0.0000003, epoch = 1000):
        """
        X: N x D
        Y: N x 1
        predict - Y: N x 1
        dlt: N x 1
        dW: 1 x D
        db: 1 x 1
        weight: D X 1
        beta: 1x 1
        """
        X = np.array(X, dtype = "float32")
        Y = np.array(Y, dtype = "float32") 

        N, D = X.shape
        Y = Y.reshape(N,1)
        
        dlt = np.zeros([N,1], dtype = "float32")
        dW = np.zeros([1,D], dtype = "float32")
        db = 0
        self.weight = np.zeros([D,1], dtype = "float32")
        self.beta = 0

        c = []
        
        for n in range(epoch):
            dlt = self.predict(X).T - Y 
            dW = np.matmul(dlt.T,X).T
            db = dlt.sum()/N

            self.weight -= learning_rate * dW
            self.beta -= learning_rate * db
            if n%1000==0:
                c_new = cost(self.predict(X).T,Y)
                c.append(c_new)
                err = error_rate(self.predict_class(X).T,Y)
                print("epoch:",n,"cost:", c_new,"error rate:",err)

##        plt.plot(c)
##        plt.show()

    def predict(self,X):
        """
        weight: D x 1
        X.T: D x N
        Output: 1 x N
        """
        z = np.matmul(self.weight.T,X.T)+self.beta 
        return 1/(1+np.exp(-z))

    def predict_class(self,X):
        predictclass = self.predict(X)
        return (predictclass >= 0.5) * 1


def main():
    X, Y = gendata()
    f, (p1, p2, p3, p4) = plt.subplots(1, 4, sharey=False)
    p1.scatter(X[:9999,0],X[:9999,1],color='blue')
    p1.scatter(X[10000:,0],X[10000:,1],color='red')
    p1.title.set_text('Original Dataset')

    model = logistic_regression()
    model.fit(X, Y,learning_rate=0.0005, epoch = 1)
    result0 = model.predict_class(X)
    ind00 = np.where(result0 == 1)
    ind01 = np.setxor1d(ind00,list(range(20000)))
    p2.scatter(X[ind00,0],X[ind00,1],color='blue')
    p2.scatter(X[ind01,0],X[ind01,1],color='red')
    p2.title.set_text('Training after 1 step')
    
    model.fit(X, Y,learning_rate=0.0005, epoch = 10000)
    result = model.predict_class(X)
    ind10 = np.where(result == 1)
    ind11 = np.setxor1d(ind10,list(range(20000)))
    p3.scatter(X[ind10,0],X[ind10,1],color='blue')
    p3.scatter(X[ind11,0],X[ind11,1],color='red')
    p3.title.set_text('Training after 10000 step')
    
    model.fit(X, Y,learning_rate=0.0005, epoch = 100000)
    result1 = model.predict_class(X)
    ind20 = np.where(result1 == 1)
    ind21 = np.setxor1d(ind20,list(range(20000)))
    p4.scatter(X[ind20,0],X[ind20,1],color='blue')
    p4.scatter(X[ind21,0],X[ind21,1],color='red')
    p4.title.set_text('Training after 100000 step')

    plt.show()
    
    Z = model.predict(X).T
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X[:,0],X[:,1], Z)
    plt.show()

if __name__ =="__main__":
    main()
