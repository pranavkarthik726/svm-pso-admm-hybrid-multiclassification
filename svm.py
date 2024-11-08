import numpy as np
import matplotlib as plt
from ADMM import *
from pso import *  
from RBFkernel import *
class SVM:
    def __init__(self, sigma=0.1, epochs=1000):
        self.Lambda = None
        self.b = 0
        self.C = 1
        self.sigma = sigma
        self.epochs = epochs
        self.kernel = gaussian_kernel  # Use the Gaussian kernel
        self.X = None  # Store training data
        self.y = None  # Store training labels
        self.pso_settings(pso_max_iter=100,pso_n_particles=10,pso_w=0.7,pso_c1=1.5,pso_c2=1.5)
    
    def pso_settings(self,pso_max_iter=100,pso_n_particles=10,pso_w=0.7,pso_c1=1.5,pso_c2=1.5,admm_max_iter=100):
        self.pso_max_iter=pso_max_iter
        self.pso_n_particles=pso_n_particles
        self.pso_w=pso_w
        self.pso_c1=pso_c1
        self.pso_c2=pso_c2
        self.admm_max_iter=admm_max_iter
    
    def train(self, X, y,admm_max_iter=1000):
        # Store training data for later use in predictions
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
        self.rho = best_params['rho']
        
        # Train final model using optimized parameters
        self.Lambda, self.b = ADMM_train(X, y, self.C, self.sigma, self.kernel,max_iter=self.admm_max_iter, rho=self.rho)
            
    def predict(self, X):
        # X: Input features for prediction
        return np.sign(self.decision_function(X))
    
    def score(self, X, y):
        # X: Input features for prediction
        # y: testing data labels
        y_hat = self.predict(X)
        return np.mean(y == y_hat)
    
    def decision_function(self, X):
        # X: Input features for prediction
        # self.X: Training data features
        # self.y: Training data labels
        return (self.Lambda * self.y).dot(self.kernel(self.X, X,self.sigma)) + self.b