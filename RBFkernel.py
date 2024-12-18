import numpy as np

def gaussian_kernel(X1, X2, sigma):
    """
    Compute the Gaussian (RBF) kernel matrix between X1 and X2.
    
    Parameters:
    -----------
    X1 : array-like of shape (n_samples_1, n_features)
        First set of samples
    X2 : array-like of shape (n_samples_2, n_features)
        Second set of samples
    sigma : float
        Kernel bandwidth parameter
    
    Returns:
    --------
    K : array-like of shape (n_samples_1, n_samples_2)
        Kernel matrix
    """
    # Compute squared Euclidean distances
    X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
    distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
    
    # Compute Gaussian kernel
    gamma = 1 / (2 * sigma**2)
    K = np.exp(-gamma * distances)
    
    return K

def sigmoid_kernel(X1, X2, gamma=1.0, coef0=0.0):
    """
    Compute the Sigmoid kernel matrix between X1 and X2.
    
    Parameters:
    -----------
    X1 : array-like of shape (n_samples_1, n_features)
        First set of samples
    X2 : array-like of shape (n_samples_2, n_features)
        Second set of samples
    gamma : float, default=1.0
        Scale parameter for the dot product
    coef0 : float, default=0.0
        Independent term in the tanh function
    
    Returns:
    --------
    K : array-like of shape (n_samples_1, n_samples_2)
        Kernel matrix
    """
    K = np.dot(X1, X2.T)
    K *= gamma
    K += coef0
    K = np.tanh(K)
    
    return K