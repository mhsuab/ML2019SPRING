import numpy as np
import pandas as pd
import sys
from numpy.linalg import inv

def generative(xfile, yfile, tfile, prefile):
    xtrain, ytrain = pd.read_csv(xfile).values, (pd.read_csv(yfile).values)
    xtest = pd.read_csv(tfile).values
    w, b = find_theta(xtrain, ytrain)
    y = func(xtest, w, b)
    with open(prefile, 'w') as f:
        f.write('id,label\n')
        for i in range(len(y)):
            f.write(str(i + 1) + ',' + str(to_01(y[i][0])) + '\n')

def find_theta(x, y):
    class_0_id = np.nonzero(y == 0)[0]
    class_1_id = np.nonzero(y)[0]

    class_0 = x[class_0_id]
    class_1 = x[class_1_id]

    mean_0 = np.mean(class_0, axis = 0)
    mean_1 = np.mean(class_1, axis = 0)  

    n = class_0.shape[1]
    cov_0 = np.zeros((n,n))
    cov_1 = np.zeros((n,n))
    
    for i in range(class_0.shape[0]):
        cov_0 += np.dot(np.transpose([class_0[i] - mean_0]), [(class_0[i] - mean_0)]) / class_0.shape[0]

    for i in range(class_1.shape[0]):
        cov_1 += np.dot(np.transpose([class_1[i] - mean_1]), [(class_1[i] - mean_1)]) / class_1.shape[0]

    cov = (cov_0 * class_0.shape[0] + cov_1*class_1.shape[0]) / (class_0.shape[0] + class_1.shape[0])

    w = np.transpose(((mean_0 - mean_1)).dot(inv(cov)))
    b =  (- 0.5)* (mean_0).dot(inv(cov)).dot(mean_0)+ 0.5 * (mean_1).dot(inv(cov)).dot(mean_1)+ np.log(float(class_0.shape[0]) / class_1.shape[0]) 
    return w, b

def func(x, w, b):
    arr = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        z = x[i,:].dot(w) + b
        arr[i][0] = 1 / (1 + np.exp(-z))
    return np.clip(arr, 1e-8, 1-(1e-8))

def to_01(y):
    return (y <= 0.5).astype(int)

if __name__ == '__main__':
    generative(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])