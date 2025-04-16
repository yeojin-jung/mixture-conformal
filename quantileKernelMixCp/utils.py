import numpy as np
from numpy.linalg import norm

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split

from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment

import ternary
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import cvxpy as cp

from tqdm import tqdm

def kernel(x, y, gamma):
    return rbf_kernel(x,y, gamma=gamma)

def pinball(beta0, y0, tau):
    """
    defines pinball loss
    """
    tmp = y0-beta0
    loss = np.sum(tau*tmp[tmp>0])+np.sum((tau-1)*tmp[tmp<=0])
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

def row_standardize(matrix):
    row_means = np.mean(matrix, axis=1, keepdims=True)
    row_stds = np.std(matrix, axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1.0
    standardized_matrix = (matrix - row_means) / row_stds
    return standardized_matrix

def run_plsi(X, k):
    U, L, V = svds(X, k)
    V  = V.T
    L = np.diag(L)
    J, H_hat = preconditioned_spa(U, k)

    W_hat = get_W_hat(U, H_hat)
    A_hat = get_A_hat(W_hat,X)      
    return W_hat, A_hat

def preprocess_U(U, K):
    for k in range(K):
        if U[0, k] < 0:
            U[:, k] = -1 * U[:, k]
    return U

def precondition_M(M, K):
    Q = cp.Variable((K, K), symmetric=True)
    objective = cp.Maximize(cp.log_det(Q))
    constraints = [cp.norm(Q @ M, axis=0) <= 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    Q_value = Q.value
    return Q_value

def preconditioned_spa(U, K, precondition=True):
    J = []
    M = preprocess_U(U, K).T
    if precondition:
        L = precondition_M(M, K)
        S = L @ M
    else:
        S = M
    
    for t in range(K):
            maxind = np.argmax(norm(S, axis=0))
            s = np.reshape(S[:, maxind], (K, 1))
            S1 = (np.eye(K) - np.dot(s, s.T) / norm(s) ** 2).dot(S)
            S = S1
            J.append(maxind)
    H_hat = U[J, :]
    return J, H_hat

def get_W_hat(U, H):
    projector = H.T.dot(np.linalg.inv(H.dot(H.T)))
    theta = U.dot(projector)
    theta_simplex_proj = np.array([_euclidean_proj_simplex(x) for x in theta])
    return theta_simplex_proj

def get_A_hat(W_hat, M):
    projector = (np.linalg.inv(W_hat.T.dot(W_hat))).dot(W_hat.T)
    theta = projector.dot(M)
    theta_simplex_proj = np.array([_euclidean_proj_simplex(x) for x in theta])
    return theta_simplex_proj

def _euclidean_proj_simplex(v, s=1):
    (n,) = v.shape
    if v.sum() == s and np.alltrue(v >= 0):
        return v
    
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = (v - theta).clip(min=0)
    return w

def get_component_mapping(stats_1, stats_2):
    similarity = stats_1.T @ stats_2
    cost_matrix = -similarity
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    P = np.zeros_like(cost_matrix)
    P[row_ind, col_ind] = 1
    return P

def plot_ternary(data_points, cover_vector, title, ax):
    scale = 1  # Simplex sum should be 1
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)

    for i, (point, cover) in enumerate(zip(data_points, cover_vector)):
        color = 'red' if cover else 'blue'  # True = Red, False = Blue
        tax.scatter([point], marker='o', color=color, s=50)

    # Configure ternary plot
    tax.boundary(linewidth=1.5)  # Draw the simplex boundary
    tax.gridlines(multiple=0.2, color="gray", linestyle="dotted")  # Grid
    tax.left_axis_label("Component 1", fontsize=12)
    tax.right_axis_label("Component 2", fontsize=12)
    tax.bottom_axis_label("Component 3", fontsize=12)
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1, offset=0.02)
    tax.clear_matplotlib_ticks()  # Remove extra ticks
    ax.set_title(title, fontsize=14)
    
def plot_ternary_size(data_points, cover_vector, title, ax,
                      vmin, vmax,
                      cmap = cm.plasma):
    scale = 1
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    scatter = tax.scatter(data_points, marker='o', c=cover_vector, 
                          cmap=cmap, s=50, vmin=vmin, vmax=vmax)

    tax.boundary(linewidth=1.5) 
    tax.gridlines(multiple=0.2, color="gray", linestyle="dotted")
    tax.left_axis_label("Component 1", fontsize=12)
    tax.right_axis_label("Component 2", fontsize=12)
    tax.bottom_axis_label("Component 3", fontsize=12)
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1, offset=0.02)
    tax.clear_matplotlib_ticks()
    ax.set_title(title, fontsize=14)

    return scatter


def wsc(X, cover, delta=0.1, M=1000, random_state=2020, verbose=False):

    # Set up the default_rng generator using the extracted seed
    rng = np.random.default_rng(random_state)

    def wsc_v(X, cover, delta, v):
        n = np.shape(X)[0]
        #cover = np.array([ r[i] <= q[i] for i in range(n)])
        z = np.dot(X,v)

        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]


        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0-delta)*n))
        ai_best = 0
        bi_best = n-1
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai+int(np.round(delta*n)),n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1,n-ai+1)
            coverage[np.arange(0,bi_min-ai)]=1
            bi_star = ai+np.argmin(coverage)
            cover_star = coverage[bi_star-ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best]

    def sample_sphere(n, p):
        v = rng.normal(size=(p, n))
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    if verbose:
        for m in tqdm(range(M)):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, cover, delta, V[m])
    else:
        for m in range(M):
            wsc_list[m], a_list[m], b_list[m] = wsc_v(X, cover, delta, V[m])
        
    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star




def wsc_unbiased(X, cover, delta=0.1, M=1000, test_size=0.8, random_state=2020, verbose=False):
    
    state = np.random.get_state()

    # Restore the original random state
    extracted_seed = state[1][0]
    
    def wsc_vab(X, cover, v, a, b):
        n = np.shape(X)[0]
        #cover = np.array([ r[i] <= q[i] for i in range(n)])
        z = np.dot(X,v)
        idx = np.where((z>=a)*(z<=b))
        coverage = np.mean(cover[idx])
        return coverage

    X_train, X_test, cover_train, cover_test = train_test_split(X, cover, test_size=test_size, random_state=extracted_seed)
    # Find adversarial parameters
    wsc_star, v_star, a_star, b_star = wsc(X_train, cover_train, delta=delta, M=M, random_state=extracted_seed, verbose=verbose)
    # Estimate coverage
    coverage = wsc_vab(X_test, cover_test, v_star, a_star, b_star)
    return coverage