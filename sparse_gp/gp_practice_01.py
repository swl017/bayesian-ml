# http://krasserm.github.io/2018/03/19/gaussian-processes/

import numpy as np
from gaussian_processes_util import plot_gp
import matplotlib.pyplot as plt

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    '''
    Isotropic squared exponential kernel. Computes 
    a covariance matrix from points in X1 and X2.
        
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).
    '''
    sqdist = np.sum(X1**2, 1).reshape(-1,1) + np.sum(X2**2,1) - 2*np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / 1**2 * sqdist)



#############33

# Prediction from noise-free training data

from numpy.linalg import inv

def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    '''  
    Computes the suffifient statistics of the GP posterior predictive distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    
    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    K     = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s   = kernel(X_train, X_s, l, sigma_f)
    K_ss  = kernel(X_s, X_s, l, sigma_f)# + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # posterior predictive distribution mean and cov
    mu_s  = K_s.T.dot(K_inv).dot(Y_train)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s



if __name__ == "__main__":

    ###################################
    ################ Prior
    
    # points
    X = np.arange(-5, 5, 0.2).reshape(-1,1)

    # mean and covariance of the PRIOR
    mu  = np.zeros(X.shape)
    cov = kernel(X, X)

    # Draw three samples from the prior
    samples = np.random.multivariate_normal(mu.ravel(), cov, 3)
    # Plot GP mean, confidence interval and samples
    # plot_gp(mu, cov, X, samples=samples)




    ####################################
    ######### Prdiction from noise-free training data

    # Noise free training data
    X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
    Y_train = np.sin(X_train)

    # Compute mean and covariance of the posterior predictive distribution
    mu_s, cov_s = posterior_predictive(X, X_train, Y_train)

    samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    # plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)




    #####################################
    ######### Prediction from noisy training data

    noise = 0.4

    # Noisy training data
    X_train = np.arange(-3, 4, 1).reshape(-1, 1)
    Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)

    # Compute mean and covariance of the posterior predictive distribution
    mu_s, cov_s = posterior_predictive(X, X_train, Y_train, sigma_y=noise)

    samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples)
    plt.show()