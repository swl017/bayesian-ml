# http://krasserm.github.io/2018/03/19/gaussian-processes/

import numpy as np
from gaussian_processes_util import plot_gp, plot_gp_compare
import matplotlib.pyplot as plt

# def mse(X_test, Y_test, X_train, Y_train):
#     for i in range(0, len(X_test)):
#         for j in range(0, len(X_train)):


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

def posterior_predictive_FITC(X_s, X_train, X_ind, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
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

        X_ind: inducing points
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    k_xx       = kernel(X_s, X_s, l, sigma_f)# + 1e-8 * np.eye(len(X_s))
    K_xXind    = kernel(X_s, X_ind, l, sigma_f)
    K_Xindx    = kernel(X_ind, X_s, l, sigma_f)
    K_XX       = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_XindX    = kernel(X_ind, X_train, l, sigma_f)
    K_XXind    = kernel(X_train, X_ind, l, sigma_f)
    K_XindXind = kernel(X_ind, X_ind, l, sigma_f)
    k_inv      = inv(K_XindXind)
    # print(K_xXind.shape, k_inv.shape, K_XindX.shape)
    Q_xX       = K_xXind.dot(k_inv).dot(K_XindX)
    Q_Xx       = K_XXind.dot(k_inv).dot(K_Xindx)
    Q_XX       = K_XXind.dot(k_inv).dot(K_XindX)
    Lambda     = np.diag(np.diag((K_XX - Q_XX + sigma_f*sigma_f*np.eye(len(X_train)))))

    Kinv = inv(Q_XX + Lambda)
    # posterior predictive distribution mean and cov
    # mu_s  = K_s.T.dot(K_inv).dot(Y_train)
    # cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    mean_z = Q_xX.dot(Kinv).dot(Y_train)
    cov_z  = k_xx - Q_xX.dot(Kinv).dot(Q_Xx)

    return mean_z, cov_z


if __name__ == "__main__":
    """
    Prior
    """

    # points
    X = np.arange(-5, 6, 0.2).reshape(-1,1)

    # mean and covariance of the PRIOR
    mu  = np.zeros(X.shape)
    cov = kernel(X, X)

    # Draw three samples from the prior
    samples = np.random.multivariate_normal(mu.ravel(), cov, 3)
    # Plot GP mean, confidence interval and samples
    # plot_gp(mu, cov, X, samples=samples)

    """
    Prediction from noisy training data
    """
    noise = 0.8
    # noise = 1e-8
    noise_assumption = 1

    # Noisy training data
    X_train = np.arange(-3, 4, 0.07).reshape(-1, 1)
    Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)
    # portion of data
    p = 10
    X_ind   = X_train[::p]
    Y_ind   = Y_train[::p]

    """Full GP"""
    # Compute mean and covariance of the posterior predictive distribution
    mu_s, cov_s = posterior_predictive(X, X_train, Y_train, sigma_y=noise_assumption)
    samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    plot_gp_compare(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples, i=1, title="Full GP with total "+str(len(X_train))+" data points")

    mu_s, cov_s = posterior_predictive(X, X_ind, Y_ind, sigma_y=noise_assumption)
    samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    plot_gp_compare(mu_s, cov_s, X, X_train=X_ind, Y_train=Y_ind, samples=samples, i=2, title="Full GP with 1/"+str(p)+" data points")

    """FITC"""
    # Compute mean and covariance of the posterior predictive distribution
    mu_z, cov_z = posterior_predictive_FITC(X, X_train, X_ind, Y_train, sigma_y=noise_assumption)

    samples = np.random.multivariate_normal(mu_z.ravel(), cov_z, 3)
    title = "FITC with 1/"+str(p)+" data points"
    plot_gp_compare(mu_z, cov_z, X, X_train=X_ind, Y_train=Y_ind, samples=samples, i=3, title=title)
    # plot_gp(mu_z, cov_z, X, X_train=X_ind, Y_train=Y_ind, samples=samples)
    plt.show()
