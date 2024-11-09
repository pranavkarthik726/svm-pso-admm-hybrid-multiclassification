import numpy as np

def ADMM_train(X, y, C, kernel, max_iter=1000, rho=1.0, tol=1e-4):
    """
    ADMM algorithm for training SVM with kernel
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target labels (+1, -1)
    C : float
        Regularization parameter
    kernel : callable
        Kernel function that takes (X1, X2) as input
    max_iter : int
        Maximum number of iterations
    rho : float
        ADMM penalty parameter
    tol : float
        Convergence tolerance
    
    Returns:
    --------
    alpha : array-like of shape (n_samples,)
        Lagrange multipliers
    b : float
        Bias term
    """
    n_samples = X.shape[0]
    
    # Initialize variables
    alpha = np.zeros(n_samples)  # Lagrange multipliers
    z = np.zeros(n_samples)      # Auxiliary variable
    u = np.zeros(n_samples)      # Scaled dual variable
    
    # Compute kernel matrix
    K = kernel(X, X)
    
    # ADMM iteration
    for iter in range(max_iter):
        # Update alpha (solve linear system)
        Q = K + (1/rho) * np.eye(n_samples)
        p = y - z + u
        alpha = np.linalg.solve(Q, p)
        
        # Update z (proximal operator)
        # ... existing code ...
        v = np.clip(alpha, -1e10, 1e10) + np.clip(u, -1e10, 1e10)  # Prevent overflow
# ... existing code ...
        z = np.clip(v, 0, C)
        
        # Update dual variable u
        u = np.clip(u + (alpha - z), -1e10, 1e10)
        
        # Check convergence
        primal_res = np.linalg.norm(alpha - z)
        dual_res = np.linalg.norm(-rho * (z - z_old) if iter > 0 else 0)
        
        if primal_res < tol and dual_res < tol:
            break
            
        z_old = z.copy()
    
    # Compute bias term b
    # Use support vectors (points where 0 < alpha < C)
    sv_idx = (alpha > 1e-6) & (alpha < C - 1e-6)
    if np.sum(sv_idx) > 0:
        b = np.mean(y[sv_idx] - np.sum(K[sv_idx][:, sv_idx] * (alpha[sv_idx] * y[sv_idx]), axis=1))
    else:
        b = 0.0
        
    return alpha, b
