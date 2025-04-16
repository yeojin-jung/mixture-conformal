import time
import sys 
import os

from quantileKernelMixCp.quikmix import QuiKMixCP
from quantileKernelMixCp.generate_qkm import *
from quantileKernelMixCp.utils import *

from conditionalconformal import CondConf
from conditionalconformal.condconf import setup_cvx_problem, finish_dual_setup
from experiments.crossval import runCV

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from PCP.utils import PCP, RLCP

def runExp(ntrials, N, n, p, p0, K, 
           test_prop = 0.1,
           calib_prop = 0.3,
           covariate_W = False,
           extra_covariate = True,
           alpha=0.1, 
           verbose=True):

    success_count = 0
    seed_index = 0
    error_count = 0
    failed_seeds = []

    coverage_df = pd.DataFrame({
        "seed": pd.Series(dtype='int'),
        "coverage": pd.Series(dtype='float'),
        "shift_type": pd.Series(dtype='int'),
        "time": pd.Series(dtype='float'),
        "cutoff": pd.Series(dtype='float'),
        "method": pd.Series(dtype='str')})

    while seed_index < ntrials:
        seed = 100 + seed_index
        np.random.seed(seed)
        print(f"🔁 Trying seed {seed} (successes: {success_count})", flush=True)

        #try:
        # === Simulate data ===
        X, y, D, W, A = generate_data(N, n, p, p0, K, test_prop, covariate_W, extra_covariate)
        splits = split_data(X, y, calib_prop, test_prop, seed)
                    
        X_train, y_train, train_idx = splits['train']
        X_calib, y_calib, calib_idx = splits['calib']
        X_test, y_test, test_idx = splits['test']

        reg = LinearRegression().fit(X_train, y_train.ravel())
        scoresCalib = np.abs(reg.predict(X_calib) - y_calib.ravel())
        scoresTest =  np.abs(reg.predict(X_test) - y_test.ravel())
        scoresTrain =  np.abs(reg.predict(X_train) - y_train.ravel())
        scoreFn = lambda x, y : np.abs(y - reg.predict(x))

        # === CondCP ===
        # default settings in the package
        k = 5
        gamma = 4
        minRad = 0.0001
        maxRad = 1
        numRad = 40
        
        start_time = time.time()
        print("Starting CondCP...")
        phiFn = lambda x : np.hstack([np.ones((len(x),1)), x[:,K:]]) if covariate_W else np.hstack([np.ones((len(x),1)), x[:,p:]])
        phiCalib = phiFn(X_calib)
        phiTest = phiFn(X_test)

        # Cross validate on lambda
        allLosses, radii = runCV(X_calib, scoresCalib, 'rbf', gamma, alpha, k,
                                    minRad, maxRad, numRad, phiCalib)
        selectedRadius = radii[np.argmin(allLosses)]
        infinite_params = {'kernel': 'rbf', 'gamma': gamma, 'lambda': 1 / selectedRadius}

        # Get coverage
        cvx_prob = setup_cvx_problem(X_calib, scoresCalib, phiCalib, infinite_params)
        covers_cc_list = []
        for i, (x_val, y_val) in enumerate(zip(X_test, scoresTest)):
            x_val = x_val.reshape(1, -1)
            prob = finish_dual_setup(cvx_prob, y_val, x_val, 1 - alpha,
                                        phiTest[i], X_calib, infinite_params)
            prob.solve(cp.MOSEK)
            eta = prob.var_dict['weights'].value
            u = np.random.uniform(-alpha,1-alpha,size=1)
            covers_cc_list.append(eta[-1] < u)
        covers_cc = np.concatenate(covers_cc_list)

        # Get cutoffs
        condCovProgram = CondConf(score_fn = scoreFn, 
                                    Phi_fn = phiFn, 
                                    infinite_params = {'kernel': 'rbf', 'gamma': gamma, 'lambda' : 1/selectedRadius})
        condCovProgram.setup_problem(X_calib, scoresCalib)

        score_inv_fn_ub = lambda s, x : [reg.predict(x) - s, reg.predict(x) + s]
        cutoffs_cc = []
        i=0
        for x_val, y_val in zip(X_test, scoresTest):
            x = x_val.reshape(1,-1)
            cutoff = condCovProgram.predict(quantile=1-alpha,
                            x_test=x,
                            score_inv_fn=score_inv_fn_ub,
                            S_min=min(scoresCalib),
                            S_max=max(scoresCalib),
                            randomize=True,
                            exact=False,
                            threshold=1-alpha)
            cutoffs_cc.append(cutoff)
            i+=1
        time_cc = time.time() - start_time


        # === QuiKMixCP ===
        print("Starting QKMCP...")
        Smin = np.min(scoresCalib)
        Smax = np.max(scoresCalib)
        start_time = time.time()
        quiKMix = QuiKMixCP(alpha = alpha, K = K, p = p, rand_seed=seed)
        quiKMix.train(X_train, X_calib, X_test, y_train, y_calib, y_test)
        covers_qkm_rand, covers_qkm = quiKMix.fit()
        covers_qkm_rand = np.concatenate(covers_qkm_rand)
        covers_qkm = np.array(covers_qkm)
        cutoffs_qkm  = quiKMix.predict(Smin, Smax)
        time_qkm = time.time()-start_time

        # === QuiKMixCP with fixed gamma ===
        print("Starting QKMCP...")
        Smin = np.min(scoresCalib)
        Smax = np.max(scoresCalib)
        start_time = time.time()
        quiKMix = QuiKMixCP(alpha = alpha, K = K, p = p, rand_seed=seed, gamma = 4)
        quiKMix.train(X_train, X_calib, X_test, y_train, y_calib, y_test)
        covers_qkm_gamma_rand, covers_qkm_gamma = quiKMix.fit()
        covers_qkm_gamma_rand = np.concatenate(covers_qkm_gamma_rand)
        covers_qkm_gamma = np.array(covers_qkm_gamma)
        cutoffs_qkm_gamma  = quiKMix.predict(Smin, Smax)
        time_qkm_gamma = time.time()-start_time

        # === SplitCP ===
        start_time = time.time()
        nCalib = len(scoresCalib)
        cutoffs_scp = np.quantile(scoresCalib, [(1 - alpha) * (1 + 1 / nCalib)])[0]
        covers_scp = (scoresTest < cutoffs_scp).astype(int)
        time_scp = time.time()-start_time

        # === PCP ===
        start_time = time.time()
        kf = KFold(n_splits=20, shuffle=True, random_state=seed)
        R_train = np.zeros_like(y_train.ravel())
        for train_index, test_index in kf.split(X_train):
            RF_train = LinearRegression().fit(X_train[train_index], y_train[train_index].ravel())
            R_train[test_index] = abs(y_train[test_index].ravel() - RF_train.predict(X_train[test_index]))

        PCP_model = PCP()
        PCP_model.train(X_train, R_train, info=True)
        cutoffs_pcp, covers_pcp = PCP_model.calibrate(X_calib, scoresCalib, X_test, scoresTest, alpha, finite=True)
        covers_pcp = np.array(covers_pcp)
        time_pcp = time.time()-start_time
        
        # === RLCP ===
        start_time = time.time()
        cutoffs_rlcp, covers_rlcp = RLCP(X_train, X_calib, scoresCalib, X_test, scoresTest, alpha, finite=True)
        covers_rlcp = np.array(covers_rlcp)
        time_rlcp = time.time()-start_time

        # === Store results ===
        methods = [
        ("QKMCP", covers_qkm, time_qkm, np.mean(cutoffs_qkm)),
        ("QKMCP_rand", covers_qkm_rand, time_qkm, np.mean(cutoffs_qkm)),
        ("QKMCP_gamma", covers_qkm_gamma, time_qkm_gamma, np.mean(cutoffs_qkm_gamma)),
        ("QKMCP_gamma_rand", covers_qkm_gamma_rand, time_qkm_gamma, np.mean(cutoffs_qkm_gamma)),
        ("PCP", covers_pcp, time_pcp, np.mean(cutoffs_pcp)),
        ("RLCP", covers_rlcp, time_rlcp, np.mean(cutoffs_rlcp)),
        ("CondCP", covers_cc, time_cc, np.mean(cutoffs_cc)),
        ("SplitCP", covers_scp, time_scp, cutoffs_scp)]

        for method, cover_array, runtime, cutoff in methods:
            raw_row = {
                "seed": seed,
                "coverage": 1 - np.mean(cover_array),
                "shift_type": "raw",
                "time": runtime,
                "cutoff": cutoff,
                "method": method
            }
            coverage_df = pd.concat([coverage_df, pd.DataFrame([raw_row])], ignore_index=True)

            for k in range(K):
                topic_weight = W[test_idx, k]
                shiftcov = np.mean(cover_array * topic_weight) / np.mean(topic_weight)
                shift_row = {
                    "seed": seed,
                    "coverage": 1 - shiftcov,
                    "shift_type": k,
                    "time": runtime,
                    "cutoff": cutoff,
                    "method": method
                }
                coverage_df = pd.concat([coverage_df, pd.DataFrame([shift_row])], ignore_index=True)

            wsc_val = wsc_unbiased(X[test_idx], cover_array)
            wsc_row = {
                "seed": seed,
                "coverage": wsc_val,
                "shift_type": "worst_slice",
                "time": runtime,
                "cutoff": cutoff,
                "method": method
            }
            coverage_df = pd.concat([coverage_df, pd.DataFrame([wsc_row])], ignore_index=True)

        print(f"✅ Success: seed {seed} → trial {success_count} done", flush=True)
        success_count += 1

        #except Exception as e:
        #    print(f"❌ Seed {seed} failed: {e}", flush=True)
        #    error_count += 1
        #    failed_seeds.append(seed)

        seed_index += 1

    print(f"\n🎉 Completed {ntrials} successful trials.")
    print(f"⚠️  {error_count} failures. Failed seeds: {failed_seeds}", flush=True)

    return coverage_df


if __name__ == "__main__":
    job_id = int(sys.argv[1])
    configs = pd.read_csv("config.txt", sep=" ")
    config = configs[configs["task_id"] == job_id]
    N = int(config["N"].iloc[0])
    n = int(config["n"].iloc[0])
    p = int(config["p"].iloc[0])
    p0 = int(config["p0"].iloc[0])
    K = int(config["K"].iloc[0])
    nsim = int(config["nsim"].iloc[0])

    del config
    results_dir = os.path.join(os.getcwd(), "output")
    if not os.path.exists(results_dir):
        try:
            os.makedirs(results_dir)
        except:
            pass
        
    results_csv_loc = os.path.join(results_dir,f'W=unknown_K={K}_p={p}_N={N}_n={n}_p0={p0}.csv')

    coverage_df = runExp(nsim, N, n, p, p0, K)

    coverage_df.to_csv(
        results_csv_loc,
        mode="a",
        header=not os.path.exists(results_csv_loc),
        index=False,
    )
    os.system(f"echo Done with experiment {job_id}!")

