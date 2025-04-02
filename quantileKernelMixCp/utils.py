import numpy as np
from numpy.linalg import norm

from scipy.sparse.linalg import svds
from scipy.optimize import linear_sum_assignment

import ternary
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import cvxpy as cp

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