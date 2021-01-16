#ifndef KERNEL_H
#define KERNEL_H

#include <memory>
#include <Eigen/Dense>

class Kernel
{
public:
    Kernel(/* args */);
    ~Kernel();

    Eigen::MatrixXd RBFKernel(Eigen::MatrixXd X1, Eigen::MatrixXd X2, double l=1.0, double sigma_f=1.0);
private:
    /* data */
};

Kernel::Kernel(/* args */)
{
}

Kernel::~Kernel()
{
}

Eigen::MatrixXd Kernel::RBFKernel(Eigen::MatrixXd X1, Eigen::MatrixXd X2, double l, double sigma_f)
{
    /*
    RBF a.k.a Isotropic squared exponential kernel.
    Computes a covariance matrix from points in X1 and X2.
        
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).

    (python)
    sqdist = np.sum(X1**2, 1).reshape(-1,1) + np.sum(X2**2,1) - 2*np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / 1**2 * sqdist)
     */
    Eigen::MatrixXd rbf_cov = Eigen::MatrixXd::Zero(X1.rows(), X2.rows());
    return rbf_cov;
}


#endif /* KERNEL_H */