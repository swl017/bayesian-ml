# http://krasserm.github.io/2018/03/19/gaussian-processes/

import numpy as np
from gaussian_processes_util import plot_gp, plot_gp_compare, plot_gp_2D
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


"""
Parameter Optimization
"""
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize

def nll_fn(X_train, Y_train, noise, naive=True):
    """
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given
    noise level.

    Args:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise level of Y_train.
        naive: if True use a naive implementation of Eq. (11), if
               False use a numerically more stable implementation.

    Returns:
        Minimization objective.
    """
    
    Y_train = Y_train.ravel()
    
    def nll_naive(theta):
        # Naive implementation of Eq. (11). Works well for the examples 
        # in this article but is numerically less stable compared to 
        # the implementation in nll_stable below.
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise**2 * np.eye(len(X_train))
        return 0.5 * np.log(det(K)) + \
               0.5 * Y_train.dot(inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2*np.pi)
        
    def nll_stable(theta):
        # Numerically more stable implementation of Eq. (11) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        
        def ls(a, b):
            return lstsq(a, b, rcond=-1)[0]
        
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise**2 * np.eye(len(X_train))
        L = cholesky(K)
        return np.sum(np.log(np.diagonal(L))) + \
               0.5 * Y_train.dot(ls(L.T, ls(L, Y_train))) + \
               0.5 * len(X_train) * np.log(2*np.pi)

    if naive:
        return nll_naive
    else:
        return nll_stable

# # Minimize the negative log-likelihood w.r.t. parameters l and sigma_f.
# # We should actually run the minimization several times with different
# # initializations to avoid local minima but this is skipped here for
# # simplicity.
# res = minimize(nll_fn(X_train, Y_train, noise), [1, 1], 
#                bounds=((1e-5, None), (1e-5, None)),
#                method='L-BFGS-B')

# # Store the optimization results in global variables so that we can
# # compare it later with the results from other implementations.
# l_opt, sigma_f_opt = res.x

# # Compute posterior mean and covariance with optimized kernel parameters and plot the results
# mu_s, cov_s = posterior(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise)
# plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
"""
Parameter Optimization end.
"""

if __name__ == "__main__":
    """
    Kernel Parameters
    """
    l = 3
    sigma_f = 1
    """
    Prior
    """

    # points
    # X = np.arange(-5, 6, 0.2).reshape(-1,1)
    rx, ry = np.arange(-5, 5, 0.3), np.arange(-5, 5, 0.3)
    gx, gy = np.meshgrid(rx, rx)

    X_2D = np.c_[gx.ravel(), gy.ravel()]
    X = X_2D
    # mean and covariance of the PRIOR
    mu  = np.zeros(X.shape)
    cov = kernel(X, X)

    # Draw three samples from the prior
    # samples = np.random.multivariate_normal(mu.ravel(), cov, 3)
    # Plot GP mean, confidence interval and samples
    # plot_gp(mu, cov, X, samples=samples)

    """
    Prediction from noisy training data
    """
    # noise = 0.8
    noise = 1e-8
    noise_assumption = 1
    noise_2D = 0.1

    # Noisy training data
    # X_train = np.arange(-3, 4, 0.07).reshape(-1, 1)
    # Y_train = np.sin(X_train) + noise * np.random.randn(*X_train.shape)
    X_2D_train = np.random.uniform(-4, 4, (100, 2))
    Y_2D_train = np.sin(0.5 * np.linalg.norm(X_2D_train, axis=1)) + \
                noise_2D * np.random.randn(len(X_2D_train))
    X_train = X_2D_train
    Y_train = Y_2D_train

    # portion of data
    p = 1
    X_ind   = X_train[::p]
    Y_ind   = Y_train[::p]

    # Parameter Optimization
    res = minimize(nll_fn(X_2D_train, Y_2D_train, noise_2D, naive=False), [l, sigma_f], 
               bounds=((1e-5, None), (1e-5, None)),
               method='L-BFGS-B')
    l_opt, sigma_f_opt = res.x
    print("Kernel parameters: l= %.2f, l_opt = %.2f, f = %.2f, f_opt = %.2f"\
        %(l, l_opt, sigma_f, sigma_f_opt))


    """Full GP"""
    # Compute mean and covariance of the posterior predictive distribution
    mu_s, cov_s = posterior_predictive(X, X_train, Y_train, l=l, sigma_f=sigma_f, sigma_y=noise_assumption)
    samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    # plot_gp_compare(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, samples=samples, i=1, title="Full GP with total "+str(len(X_train))+" data points")
    plot_gp_2D(gx, gy, mu_s, X_train, Y_train, "Full GP with total "+str(len(X_train))+" data points", 1)

    mu_s, cov_s = posterior_predictive(X, X_ind, Y_ind, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_assumption)
    # mu_s, cov_s = posterior_predictive(X, X_train, X_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise_assumption)
    samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
    # plot_gp_compare(mu_s, cov_s, X, X_train=X_ind, Y_train=Y_ind, samples=samples, i=2, title="Full GP with 1/"+str(p)+" data points")
    plot_gp_2D(gx, gy, mu_s, X_ind, Y_ind, "Full GP with 1/"+str(p)+" data points", 2)

    """FITC"""
    # Compute mean and covariance of the posterior predictive distribution
    mu_z, cov_z = posterior_predictive_FITC(X, X_train, X_ind, Y_train, l=l, sigma_f=sigma_f, sigma_y=noise_assumption)

    samples = np.random.multivariate_normal(mu_z.ravel(), cov_z, 3)
    title = "FITC with 1/"+str(p)+" data points"
    # plot_gp(mu_z, cov_z, X, X_train=X_ind, Y_train=Y_ind, samples=samples)
    # plot_gp_compare(mu_z, cov_z, X, X_train=X_ind, Y_train=Y_ind, samples=samples, i=3, title=title)
    plot_gp_2D(gx, gy, mu_s, X_ind, Y_ind, title, 3)
    plt.show()
