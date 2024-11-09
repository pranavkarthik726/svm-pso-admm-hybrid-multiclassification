import numpy as np
from ADMM import *
from RBFkernel import gaussian_kernel, sigmoid_kernel

def run_admm_and_get_accuracy(X, y, C, sigma, rho, kernel_type='rbf', kernel_params=None, admm_max_iter=1000):
    """Helper function to run ADMM and calculate accuracy"""
    if kernel_type == 'rbf':
        kernel = lambda X1, X2: gaussian_kernel(X1, X2, sigma)
    elif kernel_type == 'sigmoid':
        gamma = kernel_params.get('gamma', 1.0)
        coef0 = kernel_params.get('coef0', 0.0)
        kernel = lambda X1, X2: sigmoid_kernel(X1, X2, gamma, coef0)
    
    alpha, b = ADMM_train(X, y, C, kernel, rho=rho, max_iter=admm_max_iter)
    # Calculate predictions
    K = kernel(X, X)
    predictions = np.sign((alpha * y).dot(K) + b)
    accuracy = np.mean(predictions == y)
    return accuracy

def PSO_train(X, y, max_iter=100, n_particles=10, w=0.7, c1=1.5, c2=1.5,admm_max_iter=1000):
    """
    PSO algorithm to optimize rho, sigma, and C parameters for SVM
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target labels (+1, -1)
    max_iter : int
        Maximum number of PSO iterations
    n_particles : int
        Number of particles in swarm
    w : float
        Inertia weight
    c1, c2 : float
        Cognitive and social parameters
        
    Returns:
    --------
    best_params : dict
        Best parameters found (rho, sigma, C)
    best_fitness : float
        Best accuracy achieved
    """
    # Parameter bounds
    bounds = {
        'rho': (0.01, 20.0),    # Wider range for rho
        'sigma': (0.001, 5.0),  # Wider range for sigma
        'C': (0.1, 30.0) 
    }
    
    # Initialize particles and velocities
    particles = []
    velocities = []
    for _ in range(n_particles):
        particle = {
            # Randomly initialize rho between its lower bound (0.1) and upper bound (5.0) using uniform distribution
            'rho': np.random.uniform(bounds['rho'][0], bounds['rho'][1]),
            # Randomly initialize sigma between its lower bound (0.01) and upper bound (2.0) using uniform distribution
            'sigma': np.random.uniform(bounds['sigma'][0], bounds['sigma'][1]),
            # Randomly initialize C between its lower bound (0.1) and upper bound (10.0) using uniform distribution
            'C': np.random.uniform(bounds['C'][0], bounds['C'][1])
        }
        velocity = {
            'rho': np.random.uniform(-0.5, 0.5),
            'sigma': np.random.uniform(-0.5, 0.5),
            'C': np.random.uniform(-0.5, 0.5)
        }
        particles.append(particle)
        velocities.append(velocity)
    
    # Initialize best positions
    personal_best = particles.copy()
    personal_best_fitness = [-np.inf] * n_particles
    global_best = particles[0].copy()
    global_best_fitness = -np.inf
    
    # PSO iterations
    for iteration in range(max_iter):
        for i in range(n_particles):
            # Calculate fitness using ADMM
            try:
                current_fitness = run_admm_and_get_accuracy(
                    X, y,
                    particles[i]['C'],
                    particles[i]['sigma'],
                    particles[i]['rho'],
                    admm_max_iter=admm_max_iter
                )
                
                # Update personal best
                if current_fitness > personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = current_fitness
                    
                    # Update global best
                    if current_fitness > global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = current_fitness
                        
            except np.linalg.LinAlgError:
                # Handle numerical instability
                current_fitness = -np.inf
        
        # Update velocities and positions
        for i in range(n_particles):
            for param in ['rho', 'sigma', 'C']:
                # Update velocity
                velocities[i][param] = (
                    w * velocities[i][param] +
                    c1 * np.random.random() * (personal_best[i][param] - particles[i][param]) +
                    c2 * np.random.random() * (global_best[param] - particles[i][param])
                )
                
                # Update position
                particles[i][param] += velocities[i][param]
                
                # Apply bounds
                particles[i][param] = np.clip(
                    particles[i][param],
                    bounds[param][0],
                    bounds[param][1]
                )
        
        # Print progress
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{max_iter}, Best fitness: {global_best_fitness:.4f}")
            print(f"Best parameters: rho={global_best['rho']:.4f}, "
                  f"sigma={global_best['sigma']:.4f}, C={global_best['C']:.4f}")
    
    return global_best, global_best_fitness
    