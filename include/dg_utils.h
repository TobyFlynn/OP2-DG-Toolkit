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
  void jacobiGQ(const double alpha, const double beta, const int N,
                arma::vec &x, arma::vec &w);
  // Calculate Nth order Gauss Lobatto quadature points of Jacobi polynomial
  arma::vec jacobiGL(const double alpha, const double beta, const int N);
  // Calculate Jacobi polynomial of order N at points x
  arma::vec jacobiP(const arma::vec &x, const double alpha, const double beta,
                    const int N);
  // Calculate derivative of Jacobi polynomial of order N at points x
  arma::vec gradJacobiP(const arma::vec &x, const double alpha,
                        const double beta, const int N);
  // Calculate 2D orthonomal poly on simplex of order i,j
  arma::vec simplex2DP(const arma::vec &a, const arma::vec &b, const int i,
                       const int j);
  // Calculate derivatives of modal basis on simplex
  void gradSimplex2DP(const arma::vec &a, const arma::vec &b, const int i,
                      const int j, arma::vec &dr, arma::vec &ds);
  // Get cubature rules
  void cubature2D(const int cOrder, arma::vec &r, arma::vec &s, arma::vec &w);

  /*********************************
  * Calculating Vandermonde matrices
  **********************************/

  // Calculate 1D Vandermonde matrix
  arma::mat vandermonde1D(const arma::vec &r, const int N);
  // Calculate 2D Vandermonde matrix
  arma::mat vandermonde2D(const arma::vec &r, const arma::vec &s, const int N);
  // Vandermonde matrix for gradient of modal basis
  void gradVandermonde2D(const arma::vec &r, const arma::vec &s, const int N,
                         arma::mat &vDr, arma::mat &vDs);
  /*********************************
  * Calculating other matrices
  **********************************/

  // Calculate differentiation matrices
  void dMatrices2D(const arma::vec &r, const arma::vec &s, const arma::mat &V,
                   const int N, arma::mat &dr, arma::mat &ds);
  // Surface to volume lift matrix
  arma::mat lift2D(const arma::vec &r, const arma::vec &s,
                   const arma::uvec &fmask, const arma::mat &V, const int N);
  // Interpolation matrix
  arma::mat interpMatrix2D(const arma::vec &r, const arma::vec &s,
                           const arma::mat &invV, const int N);

  /*********************************
  * Calculating nodes
  **********************************/

  // Calculate warp function based on in interpolation nodes
  arma::vec warpFactor(const arma::vec &in, const int N);
  // Uses Warp & Blend to get optimal positions of points on an equilateral
  // triangle
  void setRefXY(const int N, arma::vec &x, arma::vec &y);
  // Convert from x-y coordinates in equilateral triangle to r-s coordinates of
  // the reference triagnle
  void xy2rs(const arma::vec &x, const arma::vec &y, arma::vec &r,
             arma::vec &s);
  // Convert from r-s coordinates to a-b coordinates
  void rs2ab(const arma::vec &r, const arma::vec &s, arma::vec &a,
             arma::vec &b);

  /*********************************
  * Misc
  **********************************/

  // Calculate number of points, number of face points from order N
  void basic_constants(const int N, int *Np, int *Nfp);
}

#endif
