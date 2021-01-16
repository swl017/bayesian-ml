/**
 * @file   sgp.cpp
 * @author Seungwook Lee, seungwook1024@gmail.com
 * @date   2021-01-16
 * @brief  Sparse Gaussian Process implementation in C++ using Eigen.
 * @ref    http://krasserm.github.io/2018/03/19/gaussian-processes/
 **/

#include "kernel.h"

using namespace Eigen;

class SGP
{
public:
    SGP(/* args */);
    ~SGP();

    void FullGPPosterior(MatrixXd X_s, MatrixXd X_train, MatrixXd Y_train, MatrixXd mu_s, MatrixXd cov_s, double l=1.0, double sigma_f=1.0, double sigma_y=1e-8);
    void FITCPosterior();

private:
    std::shared_ptr<Kernel> m_kernel_ptr;
};

SGP::SGP(/* args */)
{
    m_kernel_ptr = std::make_shared<Kernel>();
}

SGP::~SGP()
{
}

void SGP::FullGPPosterior(MatrixXd X_s, MatrixXd X_train, MatrixXd Y_train, MatrixXd mu_s, MatrixXd cov_s, double l, double sigma_f, double sigma_y)
{
    /*
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
    */
    MatrixXd K     = m_kernel_ptr->RBFKernel(X_train, X_train, l, sigma_f) + pow(sigma_y,2.0) * MatrixXd::Identity(X_train.rows(), X_train.cols());
    MatrixXd K_s   = m_kernel_ptr->RBFKernel(X_train, X_s, l, sigma_f);
    MatrixXd K_ss  = m_kernel_ptr->RBFKernel(X_s, X_s, l, sigma_f);
    MatrixXd K_inv = K.inverse();

    // Wrong expressions
    // MatrixXd mu_s  = K_s.transpose().dot(K_inv.dot(Y_train));
    // MatrixXd cov_s = K_ss - K_s.transpose().dot(K_inv.dot(K_s));
    MatrixXd mu_s  = K_s.transpose() * K_inv * Y_train;
    MatrixXd cov_s = K_ss - K_s.transpose() * K_inv * K_s;

}

int main()
{
    SGP sgp();
    return 0;
}