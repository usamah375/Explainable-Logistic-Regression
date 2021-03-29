import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy import optimize
import math
import collections

cols =  ['bias','age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

def sigmoid(z):
    if isinstance(z, (collections.Sequence, np.ndarray)) == True:
      z = np.array(z)
      g = np.zeros(z.shape)
      for i in range(len(z)):
        g[i] = (1/(1+math.exp(-z[i])))
    else:
      g = 1/(1+math.exp(-z))
    return g


def costFunction(theta, X, y):
    m = y.size
    h = np.dot(theta,np.transpose(X))
    h = sigmoid(h)
    J = 0
    grad = np.zeros(theta.shape)
    J = (-1/m)*np.sum(list(y*np.log(h)+(1-y)*np.log(1-h)))
    for i in range(len(grad)):
      grad[i] = (1/m)*np.sum(X[:,i]*(h-y))
    return J, grad

def predict(theta, X):
    m = X.shape[0] # Number of training examples
    p = np.zeros(m)
    h = np.dot(theta,np.transpose(X))
    h = sigmoid(h)
    for i in range(m):
      if h[i]>=0.5:
        p[i] = 1
      else:
        p[i] = 0
    return p

def sort(contrib,feats):
    temp = contrib.copy()
    temp.sort()
    sort_feats = []
    for ix in range(14):
        sort_feats.append(feats[contrib.index(temp[ix])])

    return sort_feats

data = pd.read_csv('Heart Disease Dataset.csv')
data = data.to_numpy()

X = data[:,:-1]
y = data[:,-1]
print(np.shape(data))

m, n = X.shape

# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X], axis=1)
print(X[1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




initial_theta = np.zeros(n+1)
options= {'maxiter': 500}
res = optimize.minimize(costFunction,initial_theta, (X, y), jac=True, method='TNC', options=options)
cost = res.fun
theta = res.x

y_pred = predict(theta,X_test)



for ix, i in enumerate(X_test):
    a = i.reshape(1,-1)
    l = []
    e = predict(theta,a)
    for j in range(14):
        l.append(a[0,j]*theta[j])
    res = sort(l,cols)
    res.remove('bias')
    if e == 1.0:
        print("Disease is predicted, three most defining features are: "+res[-1]+": "+str(a[0,cols.index(res[-1])]) + ", "+res[-2] +": "+str(a[0,cols.index(res[-2])]) +" and "+res[-3]+": "+str(a[0,cols.index(res[-2])]))
    else:
        print("Disease is not predicted, three most defining features are: "+res[0]+": "+str(a[0,cols.index(res[0])]) + ", "+res[1] +": "+str(a[0,cols.index(res[1])]) +" and "+res[2]+": "+str(a[0,cols.index(res[2])]))
    print("The true label was: " + str(y_test[ix]))
    print()

print("Overall Accuracy of the model: ",metrics.accuracy_score(y_test, y_pred))
