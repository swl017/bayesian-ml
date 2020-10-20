#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

template <int Row_X1, int Col, int Row_X2>
Matrix<double, Row_X1, Row_X2> kernel(Matrix<double, Row_X1, Col> X1, Matrix<double, Row_X2, Col> X2, double l=1.0, double sigma_f=1.0)
{
    /* Isotropic squared exponential kernel.
     * Also known as Gaussian kernel or RBF kernel. 
     * Computes a covariance matrix from points in X1 and X2.
     * 
     * Args:
     *      X1: Array of m points (m x d).
     *      X2: Array of n points (n x d).
     * 
     * Returns:
     *      Covariance matrix (m x n).
     *
     */ 
    Matrix<doube, Row_X1, Row_X2> K;
    kernel_row = Row_X1;
    kernel_col = Row_X2;
    for(int i = 0; i < kernel_row; i++){
        VectorXd xi = X1(i, all);
        for(int j = 0; j < kernel_col; j++){
            VectorXd xj = X2(j, all);
            K(i,j) = sigma_f * std::exp(- 0.5 / l*l * (xi * xj).transpose() * (xi * xj));
        }
    }
    return K;
}

template <int Row_X_s, int Row_X_train, int Row_Y_train, int Col>
void posterior_predictive(Matrix<double, Row_X_s, Col> X_s, Matrix<double, Row_X_train, Col> X_train, Matrix<double, Row_Y_train, Col> Y_train,
                          Matrix<double, Row_X_s, Col> &mu_s, Matrix<double, Row_X_s, Row_X_s> &cov_s, double l=1.0, double sigma_f=1.0, double sigma_y=1e-8)
{
    MatrixXd eye_train = MatrixXd::Identity(X_train, X_train);
    Matrix<double, Row_X_train, Row_X_train> K     = kernel(X_train, X_train, l, sigma_f) + sigma_f * sigma_f * eye_train;
    Matrix<double, Row_X_train, Row_X_s>     K_s   = kernel(X_train, X_s, l, sigma_f);
    Matrix<double, Row_X_s, Row_X_s>         K_ss  = kernel(X_s, X_s, l, sigma_f);
    Matrix<double, Row_X_train, Row_X_train> K_inv = K.inverse();

    mu_s  = K_s.transpose() * K_inv * Y_train;
    // mu_s  = K_s.transpose() * K_inv * (Y_train - mean(Z)) + mean(z); // with non-zero-mean data points
    cov_s = K_ss - K_s.transpose * K_inv * K_s;
}

