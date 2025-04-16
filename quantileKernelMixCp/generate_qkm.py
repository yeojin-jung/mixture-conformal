import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from quantileKernelMixCp.utils import *


def get_initial_centers(val, centers):
    quantiles = []
    for i in range(centers):
        quantiles.append(i * int(val.shape[0] / centers))
    return quantiles

def align_order(k, K):
    order = np.zeros(K, dtype=int)
    order[np.where(np.arange(K) != k)[0]] = np.random.choice(
        np.arange(1, K), K - 1, replace=False
    )
    order[k] = 0
    return order

def reorder_with_noise(v, order, K, r):
    u = np.random.rand()
    if u < r:
        return v[order[np.random.choice(range(K), K, replace=False)]]
    else:
        sorted_row = np.sort(v)[::-1]
        return sorted_row[order]

def sample_MN(p, N):
    return np.random.multinomial(N, p, size=1)

def generate_W(n, K):
    alpha = [5] + [1]*(K-1)
    W = np.zeros((n,K))
    probs = np.random.dirichlet(alpha, size=n)
    topics = np.random.choice(np.arange(K),n,replace=True)
    for k in range(K):
        inds = np.where(topics==k)[0]
        order = align_order(k, K)
        W[inds,:] = probs[np.ix_(inds, order)]
        
    # generate pure doc
    anchor_ind = np.random.choice(np.arange(n), K, replace=False)
    W[anchor_ind, :] = np.eye(K)
    W = np.apply_along_axis(lambda x: x/np.sum(x), 1, W)
    return W

def generate_data(N,n,p,p0,K,
                  test_prop,
                  covariate_W=False, 
                  extra_covariate=False):
    n_tc = int(n*(1-test_prop))
    W_tc  = generate_W(n_tc, K)

    # generate test mixtures with covariate shift
    alpha = [3] + [1]*(K-1)
    W_test = np.random.dirichlet(alpha, size=n-n_tc)
    n_shuffle = int(0.3 * W_test.shape[0])
    shuffle_rows = np.random.choice(W_test.shape[0], size = n_shuffle, replace=False)
    for row in shuffle_rows:
        np.random.shuffle(W_test[row])
    W = np.vstack([W_tc, W_test])
   
    A = np.random.uniform(0,1,size=(p,K))
    anchor_ind = np.random.choice(np.arange(p), K, replace=False)
    A[anchor_ind, :] = np.eye(K)
    A = np.apply_along_axis(lambda x: x/np.sum(x), 0, A)

    D0 = W @ A.T
    D = np.apply_along_axis(sample_MN, 1, D0, N).reshape(n,p)
    assert np.sum(D.sum(axis=1)!=N)==0

    X1 = D/N
    if extra_covariate:
        X0 = np.random.normal(scale=0.3, size=(n,p0))
        WX = np.hstack([W,X0])
        X = WX if covariate_W else np.hstack([X1,X0])
    else:
        WX = W
        X = W if covariate_W else X1

    n_covariate = WX.shape[1]
    beta = np.random.uniform(1,10,size=(n_covariate,1))
    beta = beta/beta.sum()
    nonlin = np.sin(2*np.pi*W[:,0])*beta[0]+np.cos(2*np.pi*W[:,1])*beta[1]+0.1*W[:,2]**2*beta[2]
    for j in range(3, K):
        nonlin += np.sin(2 * np.pi * W[:, j]) * beta[j]
    lin = np.dot(X0, beta[K:]) if extra_covariate else 0.0

    scale_1 = 0.1
    scale_2 = 0.1
    scale_3 = 0.3
    topics = np.argmax(W, axis=1)
    noise = np.random.normal(scale=np.where(topics==1, scale_1,
                                            np.where(topics == 3, scale_2, scale_3)),
                                            size=n)
    y = nonlin.reshape(n,1) + lin + noise.reshape(n,1)

    return X, y, D, W, A

def generate_reg(n, p,
                 test_prop=0.1,
                 calib_prop=0.3):
    X = np.random.normal(size=(n,p))
    beta = np.random.uniform(0,1, size = (p,1))
    beta = beta/beta.sum()
    if p > 1:
        nonlin = np.sin(3*X[:,0])*beta[0]+np.cos(2*X[:,1])*beta[1]+0.1*X[:,2]**2*beta[2]
        lin = np.dot(X[:,3:], beta[3:])
    else:
        nonlin = np.sin(2*np.pi*X)*beta/(np.pi*X)
        lin = 0.0

    noise = np.random.normal(scale = 0.05, size=(n,1))
    Y = nonlin.reshape(n,1) + lin + noise.reshape(n,1)

    ind = np.arange(len(X))
    train_idx, temp_idx = train_test_split(ind, test_size=test_prop+calib_prop, random_state=127)
    adjusted_test_prob = test_prop/(test_prop+calib_prop)
    calib_idx, test_idx = train_test_split(temp_idx, test_size=adjusted_test_prob, random_state=127)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_calib, Y_calib = X[calib_idx], Y[calib_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # Score: residual from OLS
    reg = LinearRegression().fit(X_train, Y_train.ravel())
    scoresCalib = np.abs(reg.predict(X_calib) - Y_calib.ravel())
    scoresTest =  np.abs(reg.predict(X_test) - Y_test.ravel())
    scoreFn = lambda x, y : y - reg.predict(x)

    return{
        "Xtrain": X_train,
        "Xcalib": X_calib,
        "Xtest": X_test,
        "Ytrain": Y_train,
        "Ycalib": Y_calib,
        "Ytest": Y_test,
        "scoresCalib": scoresCalib,
        "scoresTest": scoresTest
    }


def split_data(X, y, calib_prop=0.3, test_prop=0.1, random_state=127):
    n = len(X)
    n_tc = int(n*(1 - test_prop))
    test_idx = np.arange(n_tc, n)

    train_calib_idx = np.arange(n_tc)
    train_idx, calib_idx = train_test_split(
        train_calib_idx,
        test_size=calib_prop/(1-test_prop),
        random_state=random_state
    )
    data = {
        'train':   (X[train_idx],  y[train_idx],  train_idx),
        'calib':   (X[calib_idx],  y[calib_idx],  calib_idx),
        'test':    (X[test_idx],   y[test_idx],   test_idx)
    }
    return data