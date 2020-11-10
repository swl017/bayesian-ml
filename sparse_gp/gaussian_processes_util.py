import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

Y_true = []

def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))
    
    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    plt.plot(X, Y_true, lw=1, ls='--', label=f'y_true')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()

def plot_gp_compare(mu, cov, X, X_train=None, Y_train=None, samples=[], i=1, title=None):
    X = X.ravel()
    mu = mu.ravel()
    """
    Confidence level
    95% -> +-1.96   * sigma
    98% -> +-2.3263 * sigma
    99% -> +-2.5758 * sigma
    """
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    ax = plt.gcf().add_subplot(3,1,i)
    ax.set_title(title)
    ax.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.3)
    ax.plot(X, mu, label='Mean')
    ax.plot(X, Y_true, lw=1, ls='-', label=f'y_true')
    # for i, sample in enumerate(samples):
    #     ax.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        ax.plot(X_train, Y_train, 'rx')
    ax.legend()

def plot_gp_compare6(mu, cov, X, X_train=None, Y_train=None, samples=[], i=1, title=None):
    X = X.ravel()
    mu = mu.ravel()
    """
    Confidence level
    95% -> +-1.96   * sigma
    98% -> +-2.3263 * sigma
    99% -> +-2.5758 * sigma
    """
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    ax = plt.gcf().add_subplot(3,2,i)
    ax.set_title(title)
    ax.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.3)
    ax.plot(X, mu, label='Mean')
    ax.plot(X, Y_true, lw=1, ls='-', label=f'y_true')
    # for i, sample in enumerate(samples):
    #     ax.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        ax.plot(X_train, Y_train, 'rx')
    ax.legend()

def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(3, 1, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:,0], X_train[:,1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)