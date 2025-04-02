#######################################################################
### Adapted from "Quantile Regression in Reproducing Kernel Hilbert Space (Li, 2007)"
### Author: Yeo Jin Jung (yeojinjung@uchicago.edu)
### Date:    03/22/2025
#######################################################################

import numpy as np
from scipy.linalg import qr, solve
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import null_space
from quantileKernelMixCp.generate_qkm import kernel, pinball

def qKM_noPhi(X, y, a, max_steps, gamma, eps1=1e-04, eps2=1e-02):
    n, m = X.shape
    maxvars = min(m, n - 1)

    # Initialization
    ini = qKMIni_noPhi(X, y, a, gamma)
    indE, indL, indR = np.array(ini["indE"]), np.array(ini["indL"]), np.array(ini["indR"])
    u, u0 = ini["u"], ini["u0"]

    # Fixed components
    K = kernel(X, X, gamma)

    # Parameters we keep track of
    beta0 = np.zeros(max_steps + 1)
    theta = np.zeros((max_steps + 1, n))
    Cgacv = np.zeros(max_steps + 1)
    Csic = np.zeros(max_steps + 1)
    fit = np.zeros((max_steps + 1, n))
    checkf = np.zeros(max_steps + 1)
    lambda_vals = np.zeros(max_steps + 1)
    Elbow_list = [None] * (max_steps + 1)

    beta0[0], theta[0,:], lambda_vals[0] = ini["beta0"], ini["theta"], ini["lambda"]
    Elbow_list[0] = None
    fit[0, :] = beta0[0] + 1/lambda_vals[0]*K @ theta[0,]
    checkf[0] = pinball(fit[0, :], y, a)
    Cgacv[0] = checkf[0] / (n - len(indE))
    Csic[0] = np.log(checkf[0] / n) + (np.log(n) / (2 * n)) * len(indE)
    
    ### Main loop for solution path
    k = 0
    while k < max_steps:
        k += 1
        if (np.sum(indL) + np.sum(indR) == 1):
            lambda_vals[k] = 0
            #print("No points on left or right. Stopping.")
            break

        ### Event 1: Check if a point moves from R,L to Elbow
        # delta1_ = lambda[k]-lambda[k-1]
        lambd = lambda_vals[k-1]
        theta[k,:] = theta[k-1,:]

        mask_E = np.zeros(n, dtype=bool)
        mask_E[indE] = True
        notindE = np.nonzero(~mask_E)[0]
        
        gam = u0 + K[np.ix_(notindE,indE)] @ u
        fitRL =  beta0[k-1] + 1/lambd * K[notindE,:] @ theta[k-1,]
        #fk = -residual[notindE]+ y[notindE]
        delta1_vec = (fitRL - gam)/(y[notindE] - gam)

        if np.all(delta1_vec >1):
            delta1_ = np.inf
        else:
            # Determine hitting point
            pos = delta1_vec[delta1_vec <1]
            if pos.size == 0:
                delta1_ = np.inf
            else:
                tmpind = notindE[delta1_vec <1]
                istar1 = tmpind[np.argmax(pos)]
                delta1_ = lambd*(np.max(pos)-1)

                # Temporarily move point
                tmpE1 = np.append(indE, istar1)
                tmpL1, tmpR1 = indL.copy(), indR.copy()
                setstar1 = 'left' if istar1 in indL else 'right'
                if setstar1 == 'left':
                    tmpL1 = np.delete(indL, np.where(indL == istar1))
                else:
                    tmpR1 = np.delete(indR, np.where(indR == istar1))

                # Solve next linear system
                probdim = len(tmpE1) + 1
                block1 = np.hstack([np.ones((len(tmpE1),1)), K[np.ix_(tmpE1,tmpE1)]])
                block2 = np.hstack([0, np.ones(len(tmpE1))]).reshape(1,-1)
                tmpA1 = np.vstack([block1, block2])
                tmpY = np.hstack([y[tmpE1],0])
                q, r = qr(tmpA1, mode='economic')
                if np.linalg.matrix_rank(r) < probdim:
                    delta1_ = np.inf
                else:
                    tmpu1 = solve(r, q.T @ tmpY)

        ### Event 2: Check if a point leaves Elbow
        # deltaL = lambda[k+1]-lambda[k] when theta = a
        # deltaR = lambda[k+1]-lambda[k] when theta = -(1-a)
        if len(indE) > 2:
            joinL_bool = np.zeros(len(indE), dtype=bool)
            delta2_list = np.zeros(len(indE))
            for i in range(len(indE)):
                deltaL = (a - 1 - theta[k-1,i])/u[i]
                deltaR = (a - theta[k-1,i])/u[i]
                delta2 = np.array([deltaL, deltaR])
                delta2_ = np.inf if np.all(delta2>=0) else np.max(delta2[delta2<0]) 
                joinL_bool[i] = True if delta2_ == deltaL else False
                delta2_list[i] = delta2_

            if np.all(delta2_list > 0):
                delta2_ = np.inf
            else:
                # Determine leaving point
                pos = delta2_list[delta2_list<0]
                tmpind = indE[delta2_list<0]
                istar2 = tmpind[np.argmax(pos)]
                delta2_ = np.max(pos)
                joinL = joinL_bool[np.where(delta2_list == delta2_)[0][0]]

                # Temporarily move point
                tmpE2 = np.delete(indE, np.where(indE == istar2))
                tmpL2, tmpR2 = indL.copy(), indR.copy()
                setstar2 = 'left' if joinL else 'right'
                if joinL:
                    tmpL2 = np.append(indL, istar2)
                else:
                    tmpR2 = np.append(indR, istar2)

                # Solve next linear system
                probdim = len(tmpE2) + 1
                block1 = np.hstack([np.ones((len(tmpE2),1)), K[np.ix_(tmpE2,tmpE2)]])
                block2 = np.hstack([0, np.ones(len(tmpE2))])
                tmpA2 = np.vstack([block1, block2])
                tmpY = np.hstack([y[tmpE2],0])
                q, r = qr(tmpA2, mode='economic')
                if np.linalg.matrix_rank(r) < probdim:
                    delta2_ = np.inf
                else:
                    tmpu2 = solve(r, q.T @ tmpY)
        else:
            delta2_ = np.inf

        ### Which event happened?
        # Terminate if all step size > 0
        delta_list = np.array([delta1_, delta2_])
        #print(f"Delta list is {delta_list}")
        if np.all(delta_list > 0):
            delta = np.inf
            #print("Path stops here.")
            break

        ### Update parameters
        delta = np.max(delta_list[delta_list<0])
        lambda_vals[k] = max([lambda_vals[k-1] + delta,0])
        if lambda_vals[k] == 0:
            break
        if lambda_vals[k-1] - lambda_vals[k] < eps2:
            #print("Descent too small. Stopping.")
            break
        delta_r = 1+delta/lambd
        beta0[k] = 1/delta_r * beta0[k-1] + u0*(1-1/delta_r)
        theta[k,indE] = theta[k-1,indE] + u*delta
        Elbow_list[k] = indE

        ### Compute SIC and GACV
        fit[k, :] = beta0[k] + 1/lambda_vals[k]* K @ theta[k,]
        pb_loss = pinball(fit[k, :], y, a)
        Cgacv[k] = pb_loss / (n - len(indE))
        Csic[k] = np.log(pb_loss / n) + (np.log(n) / (2 * n)) * len(indE)
        checkf[k] = pb_loss

        if delta == delta1_: # a point hits elbow
            u0 = tmpu1[0]
            u = tmpu1[1:]
            indE, indL, indR = tmpE1, tmpL1, tmpR1
            #print(f"Observation {istar1} from {setstar1} hits elbow.")
        else: # a point leaves elbow
            u0 = tmpu2[0]
            u = tmpu2[1:]
            indE, indL, indR = tmpE2, tmpL2, tmpR2
            #print(f"Observation {istar2} leaves elbow and joins {setstar2}.")
    
    #opt = np.argmin(Csic)
    #lambd_opt = lambda_vals[opt]

    #print(f"Number of iterations run: {k+1}, Size of elbow: {len(indE)}, Lambda: {lambd_opt}")

    result = {
        "beta0": beta0[:k+1],
        "theta": theta[:k+1,:],
        "Elbow": Elbow_list[:k+1],
        "lambda": lambda_vals[:k+1],
        "fit": fit[:k+1],
        "Csic": Csic[:k+1],
        "Cgacv": Cgacv[:k+1],
        "indE": indE,
        "indR": indR,
        "indL": indL,
    }
    return result


def qKMIni_noPhi(X, y, a, gamma):
    n, m = X.shape
    yr = np.sort(y)
    quant = yr[int(np.floor(n * a))]
    istar = np.argmin(np.abs(y - quant)) # to treat case when y values are not distinct

    # Fixed components
    K = kernel(X, X, gamma)

    # Initialize sets
    indE = [istar]
    indR = np.where(y > y[istar])[0].tolist()
    indL = np.where(y < y[istar])[0].tolist()
    
    # For index, we need beta0+Phi(x_istar)^T*beta=quant
    theta = np.zeros(n)
    theta[indL] = -(1-a)
    theta[indR] = a

    # Find next hitting point
    theta_star = (1-a)*len(indL)-a*len(indR)
    theta[istar] = theta_star
    notindE = np.setdiff1d(range(n), indE)
    gam = a*K[np.ix_(notindE, notindE)].sum(axis=1)-K[np.ix_(notindE, indL)].sum(axis=1)+theta_star*K[notindE, istar]
    gam_star = a*K[istar, notindE].sum()-K[istar, indL].sum()+theta_star*K[istar,istar]
    denoms = y[notindE]-y[istar]
    lambdas = (gam-gam_star)/denoms
    lambd = np.max(lambdas)
    istar1 = notindE[np.argmax(lambdas)]

    beta0 = y[istar]-1/lambd*K[istar,:] @ theta

    # Solve next linear system
    tmpE = indE+[istar1]
    tmpL, tmpR = indL.copy(), indR.copy()
    if istar1 in tmpL:
        tmpL.remove(istar1)
    else:
        tmpR.remove(istar1)
    
    probdim = len(tmpE) + 1
    block1 = np.hstack([np.ones((len(tmpE),1)), K[np.ix_(tmpE,tmpE)]])
    block2 = np.hstack([0, np.ones(len(tmpE))]).reshape(1,-1)
    tmpA = np.vstack([block1, block2])
    tmpY = np.hstack([y[tmpE], 0])
    q, r = qr(tmpA, mode='economic')
    tmpu = solve(r, q.T @ tmpY)
    if np.linalg.matrix_rank(r) < probdim:
        print("System not solvable.")

    # Update parameters
    u0 = tmpu[0]
    u = tmpu[1:]

    return {
        "theta": theta,
        "beta0": beta0,
        "u": u,
        "u0": u0,
        "lambda": lambd,
        "indE": tmpE,
        "indR": tmpR,
        "indL": tmpL
    }


def qKMPredict_noPhi(X, y, a, x_test,
                 max_steps, Smin, Smax, 
                 max_iter=200, gamma=1, eps1=1e-4, eps2=1e-02, tol=1e-3):
    lambdas = []
    iter_count = 0

    x_val = x_test.reshape(1, -1)
    
    # Binary search to find the maximal S with S < fit(S)
    while (Smax - Smin) > tol and iter_count < max_iter:
        Smed = (Smax + Smin) / 2.0

        X_new = np.vstack([X, x_val])
        y_new = np.append(y, Smed)
        
        res = qKM_noPhi(X_new, y_new.ravel(), a, max_steps, gamma, eps1, eps2)
        opt = np.argmin(res['Csic'])
        score_diff = res['fit'][opt][-1] - Smed
        
        if score_diff > 0:
            Smin = Smed
        else:
            Smax = Smed
        
        lambdas.append(res['lambda'][opt])
        iter_count += 1

    return Smin, Smax, lambdas