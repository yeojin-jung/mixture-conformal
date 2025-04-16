#######################################################################
### Adapted from "Quantile Regression in Reproducing Kernel Hilbert Space (Li, 2007)"
### Author: Yeo Jin Jung (yeojinjung@uchicago.edu)
### Date:    03/22/2025
#######################################################################
import numpy as np

from quantileKernelMixCp.utils import *
from quantileKernelMixCp.qkm import qkm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from tqdm.notebook import tqdm

class QuiKMixCP:
    def __init__(
            self, 
            alpha: float, 
            K: int, 
            p: int, 
            rand_seed:int,
            Phifn=None, 
            gamma=None,
            gamma_grid=np.logspace(0,2,20), 
            max_steps = 1000,
            eps1=1e-06, 
            eps2=1e-02, 
            estimate_mixture = True,
        ):
        self.alpha = alpha
        self.K = K
        self.p = p
        if Phifn is None:
            self.Phifn = lambda x: x[:, :K]
        else:
            self.Phifn = Phifn
        self.gamma = gamma
        self.gamma_grid = gamma_grid
        self.max_steps = max_steps
        self.eps1 = eps1
        self.eps2 = eps2
        self.seed = rand_seed
        self.estimate_mixture = estimate_mixture

    def optimize_gamma(self, X, y, PhiX):
        """
        Optimizes gamma via Schwarz information criterion
        """
        csic_list = []
        lambdas_list = []
        for gamma in self.gamma_grid:
            #print(f"Gamma is {gamma}")
            res = qkm(X, y.ravel(), 
                    1-self.alpha, 
                    PhiX, 
                    self.max_steps, 
                    gamma,
                    self.eps1,
                    self.eps2)
            csic_list.append(np.min(res['Csic']))
            lambdas_list.append(res['lambda'][np.argmin(res['Csic'])])
        return csic_list, lambdas_list


    def estimate_mixtures(self, freq_train, freq_calib, freq_test):
        """
        Runs probabilistic latent semantic indexing (pLSI) to 
        estimate latent factors (A) and mixture proportions (W)
        """
        freq = np.vstack([freq_train, freq_calib, freq_test])
        W_hat, A_hat = run_plsi(freq, self.K)

        n_train = freq_train.shape[0]
        n_calib = freq_calib.shape[0]

        W_train = W_hat[:n_train, :]
        W_calib = W_hat[n_train:n_train + n_calib, :]
        W_test = W_hat[n_train + n_calib:, :]
        #W_train, A_train = run_plsi(freq_train, self.K)

        #W_calib, A_calib = run_plsi(freq_calib, self.K)
        #P_calib = get_component_mapping(A_calib.T, A_train.T)
        #W_calib_aligned = W_calib @ P_calib.T

        #W_test, A_test = run_plsi(freq_test, self.K)
        #P_test = get_component_mapping(A_test.T, A_train.T)
        #W_test_aligned = W_test @ P_test.T

        return W_train, W_calib, W_test


    def train(self, X_train, X_calib, X_test, y_train, y_calib, y_test):
        """
        Trains predictor, estimate latent mixtures, and 
        search over gamma grid for the optimal gamma
        """
        cv_idx, model_idx = train_test_split(
            np.arange(len(X_train)),
            test_size=0.5,
            random_state=self.seed
        )

        # Step 1: Train predictor
        reg = LinearRegression().fit(X_train[model_idx], y_train[model_idx].ravel())
        self.scoresCalib = np.abs(reg.predict(X_calib) - y_calib.ravel())
        self.scoresTest =  np.abs(reg.predict(X_test) - y_test.ravel())
        self.scoresTrain = np.abs(reg.predict(X_train[cv_idx]) - y_train[cv_idx].ravel())
        
        # Step 2: Estimate latent mixture proportions
        if self.estimate_mixture:
            self.W_train, self.W_calib, self.W_test = self.estimate_mixtures(
                                                        X_train[cv_idx,:self.p],
                                                        X_calib[:,:self.p],
                                                        X_test[:,:self.p])
        else:
            self.W_train, self.W_calib, self.W_test = X_train[cv_idx,:self.p], X_calib[:,:self.p],X_test[:,:self.p]
        # clr transformation since the data is compositional
        self.X_train_clr = np.apply_along_axis(clr, 1, self.W_train)
        self.X_calib_clr = np.apply_along_axis(clr, 1, self.W_calib)
        self.X_test_clr = np.apply_along_axis(clr, 1, self.W_test)

        p0 = X_train.shape[1]
        if self.p < p0: # there are extra covariates
            self.X_train_clr = np.hstack([self.X_train_clr, X_train[cv_idx,self.p:]])
            self.X_calib_clr = np.hstack([self.X_calib_clr, X_calib[:,self.p:]])
            self.X_test_clr = np.hstack([self.X_test_clr, X_test[:,self.p:]])

            self.X_train_clr = row_standardize(self.X_train_clr)
            self.X_calib_clr = row_standardize(self.X_calib_clr)
            self.X_test_clr = row_standardize(self.X_test_clr)

        # Step 3: Optimize for gamma in the gaussian kernel
        if self.gamma is None:
            Phi_train = self.Phifn(self.W_train)
            csic_list, _ = self.optimize_gamma(self.X_train_clr, self.scoresTrain.ravel(), Phi_train)
            self.gamma = self.gamma_grid[np.argmin(csic_list)]
            print(f"Optimal gamma is {self.gamma}.")

        

    def fit(self):
        """
        Get coverage for each test point
        """
        Phi_calib = self.Phifn(self.W_calib)
        Phi_test = self.Phifn(self.W_test)
        
        covers_rand = []
        covers = []
        for m, (x_val, y_val) in enumerate(zip(self.X_test_clr, self.scoresTest)):
            x_val = x_val.reshape(1, -1)
            X = np.vstack([self.X_calib_clr, x_val])
            y = np.append(self.scoresCalib, y_val)
            PhiX = np.vstack([Phi_calib, Phi_test[m].reshape(1, -1)])
            res = qkm(X, y.ravel(), 
                    1-self.alpha, 
                    PhiX, 
                    self.max_steps, 
                    self.gamma,
                    self.eps1,
                    self.eps2)
            opt = np.argmin(res['Csic'])
            theta_est = res['theta'][opt]
            u = np.random.uniform(-self.alpha,1-self.alpha,size=1)
            covers_rand.append(theta_est[-1] < u)
            covers.append(theta_est[-1] < 1-self.alpha)
        return covers_rand, covers


    def predict(self, Smin, Smax, max_iter=100, tol=1e-2):
        Phi_calib = self.Phifn(self.W_calib)
        Phi_test = self.Phifn(self.W_test)

        all_lengths = []
        init_Smin, init_Smax = Smin, Smax

        for m, x_val in tqdm(enumerate(self.X_test_clr), total=len(self.X_test_clr), desc="Predicting"):
            current_Smin, current_Smax = init_Smin, init_Smax
            iter_count = 0

            x_val = x_val.reshape(1, -1)
            X = np.vstack([self.X_calib_clr, x_val])
            PhiX = np.vstack([Phi_calib, Phi_test[m].reshape(1, -1)])

            while (current_Smax - current_Smin) > tol and iter_count < max_iter:
                Smed = (current_Smax + current_Smin) / 2.0
                y = np.append(self.scoresCalib, Smed)
                res = qkm(X, y.ravel(), 
                        1 - self.alpha, 
                        PhiX, 
                        self.max_steps, 
                        self.gamma,
                        self.eps1,
                        self.eps2)
                opt = np.argmin(res['Csic'])
                score_diff = res['fit'][opt][-1] - Smed

                if score_diff > 0:
                    current_Smin = Smed
                else:
                    current_Smax = Smed

                iter_count += 1

            all_lengths.append(current_Smin)
        return all_lengths