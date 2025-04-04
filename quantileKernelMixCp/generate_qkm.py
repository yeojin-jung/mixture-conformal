import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from quantileKernelMixCp.utils import *


def kernel(x, y, gamma):
    return rbf_kernel(x,y, gamma=gamma)

def pinball(beta0, y0, tau):
    tmp = y0-beta0
    loss = np.sum(tau*tmp[tmp>0])-np.sum((1-tau)*tmp[tmp<=0])
    return loss

def clr(probs):
    continuous = np.log(probs + np.finfo(probs.dtype).eps)
    continuous -= continuous.mean(-1, keepdims=True)
    return continuous

def alr(probs):
    probs = probs.copy()
    probs /= probs[-1]
    #continuous = np.log(probs + np.finfo(probs.dtype).eps)
    return probs[:-1]

def clr_then_stack(data, K, p0):
    X1_clr = np.apply_along_axis(clr, 1, data[:,:K])
    X_clr = np.hstack([X1_clr,data[:,K:K+p0]])
    return X_clr

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

def generate_data(N,n,p,p0,K,test_prop,calib_prop,covariate_W=False):
    W  = generate_W(n, K)
    A = np.random.uniform(0,1,size=(p,K))
    anchor_ind = np.random.choice(np.arange(p), K, replace=False)
    A[anchor_ind, :] = np.eye(K)
    A = np.apply_along_axis(lambda x: x/np.sum(x), 0, A)

    D0 = W @ A.T
    D = np.apply_along_axis(sample_MN, 1, D0, N).reshape(n,p)
    assert np.sum(D.sum(axis=1)!=N)==0

    X1 = D/N
    X0 = np.random.normal(scale=0.3, size=(n,p0))

    WX = np.hstack([W,X0])
    X = WX if covariate_W else np.hstack([X1,X0])

    n_covariate = WX.shape[1]
    beta = np.random.uniform(1,10,size=(n_covariate,1))
    beta = beta/beta.sum()
    nonlin = np.sin(3*W[:,0])*beta[0]+np.cos(2*W[:,1])*beta[1]+0.1*W[:,2]**2*beta[2]
    for j in range(3, K):
        nonlin += np.sin(2 * np.pi * W[:, j]) * beta[j]
    lin = np.dot(X0, beta[K:])

    scale_1 = 0.05
    scale_2 = 0.05
    scale_3 = 0.05
    topics = np.argmax(W, axis=1)
    noise = np.random.normal(scale=np.where(topics==1, scale_1,
                                            np.where(topics == 3, scale_2, scale_3)),
                                            size=n)
    Y = nonlin.reshape(n,1) + lin + noise.reshape(n,1)
    #Y = lin + noise.reshape(n,1)

    ind = np.arange(len(X))
    train_idx, temp_idx = train_test_split(ind, test_size=test_prop+calib_prop, random_state=127)
    adjusted_test_prob = test_prop/(test_prop+calib_prop)
    calib_idx, test_idx = train_test_split(temp_idx, test_size=adjusted_test_prob, random_state=127)

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_calib, Y_calib = X[calib_idx], Y[calib_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    return X_train, X_calib, X_test, Y_train, Y_calib, Y_test, train_idx, calib_idx, test_idx, D, W, A


def generate_reg(n, p,
                 test_prop=0.1,
                 calib_prop=0.3):
    X = np.random.uniform(size=(n,p))
    beta = np.random.uniform(1,10, size = (p,1))
    beta = beta/beta.sum()
    nonlin = np.sin(3*X[:,0])*beta[0]+np.cos(2*X[:,1])*beta[1]+0.1*X[:,2]**2*beta[2]
    lin = np.dot(X[:,3:], beta[3:])

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
        "Xcalib": X_calib,
        "Xtest": X_test,
        "Ycalib": Y_calib,
        "Ytest": Y_test,
        "scoresCalib": scoresCalib,
        "scoresTest": scoresTest
    }

def generate_qkm(N,n,p,p0,K,
                 test_prop=0.1,
                 calib_prop=0.3,
                 covariate_W=True):
    #np.random.seed(100)
    X_train, X_calib, X_test, Y_train, Y_calib, Y_test, train_idx, calib_idx, test_idx, D, W, A = generate_data(N,
                                                                            n,p,p0,K,test_prop,
                                                                            calib_prop, 
                                                                            covariate_W)
    # Covariate for kernel: [clr(W), X0], dim: n x K+p0
    X_train_clr = clr_then_stack(X_train, K,p0)
    X_calib_clr = clr_then_stack(X_calib, K, p0)
    X_test_clr = clr_then_stack(X_test, K, p0)

    # Score: residual from OLS
    reg = LinearRegression().fit(X_train_clr, Y_train.ravel())
    scoresCalib = np.abs(reg.predict(X_calib_clr) - Y_calib.ravel())
    scoresTest =  np.abs(reg.predict(X_test_clr) - Y_test.ravel())
    scoreFn = lambda x, y : y - reg.predict(x)

    # Covariate for Phi, dim: n x K-1
    phiCalib = X_calib[:, :K]
    phiTest = X_test[:, :K]
    phiFn = lambda x : x[:, :K]

    return{
        "Xcalib": X_calib,
        "Xtest": X_test, 
        "Xcalib_clr": X_calib_clr,
        "Xtest_clr": X_test_clr,
        "Ycalib": Y_calib,
        "Ytest": Y_test,
        "scoresCalib": scoresCalib,
        "scoresTest": scoresTest,
        "phiCalib": phiCalib,
        "phiTest": phiTest,
        "train_idx": train_idx,
        "calib_idx": calib_idx,
        "test_idx": test_idx,
        "D": D,
        "W":W,
        "A":A
    }


def generate_qkmTM(N,n,p,p0,K,
                 test_prop=0.1,
                 calib_prop=0.3,
                 covariate_W=False):
    #np.random.seed(100)
    X_train, X_calib, X_test, Y_train, Y_calib, Y_test, train_idx, calib_idx, test_idx, D, W, A = generate_data(N,
                                                                            n,p,p0,K,test_prop,
                                                                            calib_prop, 
                                                                            covariate_W)
    
    # Run topic model
    freq_train = X_train[:,:p]
    W_train, A_train = run_plsi(freq_train, K)
    WX0_train = np.hstack([W_train, X_train[:,p:]])

    freq_calib = X_calib[:,:p]
    W_calib, A_calib = run_plsi(freq_calib, K)
    P_calib = get_component_mapping(A_calib.T, A_train.T)
    W_calib_aligned = W_calib @ P_calib.T
    WX0_calib = np.hstack([W_calib_aligned, X_calib[:,p:]])

    freq_test = X_test[:,:p]
    W_test, A_test = run_plsi(freq_test, K)
    P_test = get_component_mapping(A_test.T, A_train.T)
    W_test_aligned = W_test @ P_test.T
    WX0_test = np.hstack([W_test_aligned, X_test[:,p:]])

    # Covariate for kernel: [clr(W), X0], dim: n x K+p0
    X_train_clr = clr_then_stack(WX0_train, K,p0)
    X_calib_clr = clr_then_stack(WX0_calib, K, p0)
    X_test_clr = clr_then_stack(WX0_test, K, p0)

    # ScoreTM: residual from OLS with estimated W
    reg = LinearRegression().fit(X_train_clr, Y_train.ravel())
    scoresTMCalib = np.abs(reg.predict(X_calib_clr) - Y_calib.ravel())
    scoresTMTest =  np.abs(reg.predict(X_test_clr) - Y_test.ravel())

    # Score: residual from OLS
    reg = LinearRegression().fit(X_train, Y_train.ravel())
    scoresCalib = np.abs(reg.predict(X_calib) - Y_calib.ravel())
    scoresTest =  np.abs(reg.predict(X_test) - Y_test.ravel())

    # Covariate for Phi, dim: n x K-1
    phiCalib = W_calib_aligned[:, :K-1]
    phiTest = W_test_aligned[:, :K-1]

    return{
        "Xcalib": X_calib_clr,
        "Xtest": X_test_clr,
        "Ycalib": Y_calib,
        "Ytest": Y_test,
        "scoresTMCalib": scoresTMCalib,
        "scoresTMTest": scoresTMTest,
        "scoresCalib": scoresCalib,
        "scoresTest": scoresTest,        
        "phiCalib": phiCalib,
        "phiTest": phiTest,
        "train_idx": train_idx,
        "calib_idx": calib_idx,
        "test_idx": test_idx,
        "D": D,
        "W":W,
        "A":A
    }
