#ifndef __DG_UTILS_H
#define __DG_UTILS_H

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include <vector>

namespace DGUtils {

  const double PI = 3.141592653589793238463;

  /*********************************
  * Calculating polynomials
  **********************************/

  // Calculate Nth order Gauss quadature points and weights of Jacobi polynomial
  void jacobiGQ(const double alpha, const double beta, const int n,
                arma::vec &x, arma::vec &w);
  // Calculate Nth order Gauss Lobatto quadature points of Jacobi polynomial
  arma::vec jacobiGL(const double alpha, const double beta, const int N);

  /*********************************
  * Calculating Vandermonde matrices
  **********************************/

  // Calculate 1D Vandermonde matrix
  arma::mat vandermonde1D(const arma::vec &r, const int N);

  /*********************************
  * Calculating nodes
  **********************************/

  // Calculate warp function based on in interpolation nodes
  std::vector<double> warpFactor(std::vector<double> in, const int N);
  // Uses Warp & Blend to get optimal positions of points on the 'model'
  // equilateral triangle element
  void setXY(std::vector<double> &x, std::vector<double> &y, const int N);

  /*********************************
  * Misc
  **********************************/

  // Calculate number of points, number of face points from order N
  void basic_constants(const int N, int *Np, int *Nfp);
}

#endif
