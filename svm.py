import numpy as np
import matplotlib as plt
from ADMM import *
from pso import *  
from RBFkernel import *
class SVM:
    def __init__(self, kernel_type='rbf', sigma=0.1, gamma=1.0, coef0=0.0, epochs=1000):
        self.Lambda = None
        self.b = 0
        self.C = 1
        self.sigma = sigma
        self.epochs = epochs
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.coef0 = coef0
        self.X = None
        self.y = None
        self.pso_settings(pso_max_iter=100, pso_n_particles=10, pso_w=0.7, pso_c1=1.5, pso_c2=1.5)
    
    def pso_settings(self,pso_max_iter=100,pso_n_particles=10,pso_w=0.7,pso_c1=1.5,pso_c2=1.5,admm_max_iter=100):
        self.pso_max_iter=pso_max_iter
        self.pso_n_particles=pso_n_particles
        self.pso_w=pso_w
        self.pso_c1=pso_c1
        self.pso_c2=pso_c2
        self.admm_max_iter=admm_max_iter
    
    def get_kernel(self):
        """Returns the appropriate kernel function based on kernel_type"""
        if self.kernel_type == 'rbf':
            return lambda X1, X2: gaussian_kernel(X1, X2, self.sigma)
        elif self.kernel_type == 'sigmoid':
            return lambda X1, X2: sigmoid_kernel(X1, X2, self.gamma, self.coef0)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
    
    def train(self, X, y, admm_max_iter=1000):
        self.X = X
        self.y = y
        
        # Train using PSO to find optimal parameters
        best_params, best_fitness = PSO_train(
            X, y, 
            max_iter=self.pso_max_iter,
            n_particles=self.pso_n_particles,
            w=self.pso_w,
            c1=self.pso_c1,
            c2=self.pso_c2,
            admm_max_iter=self.admm_max_iter
        )
        
        # Update parameters with PSO-optimized values
        self.C = best_params['C']
        self.sigma = best_params['sigma']

        self.gamma = best_params['sigma']
        self.rho = best_params['rho']
        
        # Train final model using optimized parameters
        kernel = self.get_kernel()
        self.Lambda, self.b = ADMM_train(X, y, self.C, kernel, max_iter=admm_max_iter, rho=self.rho)
    
    def predict(self, X):
        return np.sign(self.decision_function(X))
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y == y_hat)
    
    def decision_function(self, X):
        kernel = self.get_kernel()
        return (self.Lambda * self.y).dot(kernel(self.X, X)) + self.b