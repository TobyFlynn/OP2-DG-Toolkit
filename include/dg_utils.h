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
  // Calculate 3D orthonomal poly on simplex of order i,j,k
  arma::vec simplex3DP(const arma::vec &a, const arma::vec &b,
                       const arma::vec &c, const int i, const int j,
                       const int k);
  // Calculate derivatives of modal basis on simplex
  void gradSimplex2DP(const arma::vec &a, const arma::vec &b, const int i,
                      const int j, arma::vec &dr, arma::vec &ds);
  // Calculate the gradient of the 3D orthonomal poly on simplex of order i,j,k
  void gradSimplex3DP(const arma::vec &a, const arma::vec &b,
                      const arma::vec &c, const int i, const int j, const int k,
                      arma::vec &dr, arma::vec &ds, arma::vec &dt);
  // Get cubature rules
  void cubature2D(const int cOrder, arma::vec &r, arma::vec &s, arma::vec &w);

  /*********************************
  * Calculating Vandermonde matrices
  **********************************/

  // Calculate 1D Vandermonde matrix
  arma::mat vandermonde1D(const arma::vec &r, const int N);
  // Calculate 2D Vandermonde matrix
  arma::mat vandermonde2D(const arma::vec &r, const arma::vec &s, const int N);
  // Caclulate 3D Vandermonde matrix
  arma::mat vandermonde3D(const arma::vec &r, const arma::vec &s,
                          const arma::vec &t, const int N);
  // Vandermonde matrix for gradient of modal basis
  void gradVandermonde2D(const arma::vec &r, const arma::vec &s, const int N,
                         arma::mat &vDr, arma::mat &vDs);
  // Caclulate 3D gradient Vandermonde matrices
  void gradVandermonde3D(const arma::vec &r, const arma::vec &s,
                         const arma::vec &t, const int N, arma::mat &dr,
                         arma::mat &ds, arma::mat &dt);
  /*********************************
  * Calculating other matrices
  **********************************/

  // Calculate differentiation matrices
  void dMatrices2D(const arma::vec &r, const arma::vec &s, const arma::mat &V,
                   const int N, arma::mat &dr, arma::mat &ds);
  // Calculate differentiation matrices
  void dMatrices3D(const arma::vec &r, const arma::vec &s, const arma::vec &t,
                   const arma::mat &V, const int N, arma::mat &dr,
                   arma::mat &ds, arma::mat &dt);
  // Surface to volume lift matrix
  arma::mat lift2D(const arma::vec &r, const arma::vec &s,
                   const arma::uvec &fmask, const arma::mat &V, const int N);
  // Calculate 3D surface to volume lift operator
  arma::mat lift3D(const arma::vec &r, const arma::vec &s, const arma::vec &t,
                   const arma::uvec &fmask, const arma::mat &v, const int N);
  // Interpolation matrix
  arma::mat interpMatrix2D(const arma::vec &r, const arma::vec &s,
                           const arma::mat &invV, const int N);
  // Calculate interpolation matrices
  arma::mat interpMatrix3D(const arma::vec &r, const arma::vec &s,
                           const arma::vec &t, const arma::mat &invV,
                           const int N);
  // Calculate the mass matrix of each face
  void faceMassMatrix3D(const arma::vec &r, const arma::vec &s,
                        const arma::vec &t, const arma::uvec &fmask,
                        const arma::mat &v, const int N, arma::mat &face0,
                        arma::mat &face1, arma::mat &face2, arma::mat &face3);

  /*********************************
  * Calculating nodes
  **********************************/

  // Calculate warp function based on in interpolation nodes
  arma::vec warpFactor(const arma::vec &in, const int N);
  // Uses Warp & Blend to get optimal positions of points on an equilateral
  // triangle
  void setRefXY(const int N, arma::vec &x, arma::vec &y);
  // Uses Warp & Blend to get optimal positions of points on a reference
  // tetrahedron
  void setRefXYZ(const int N, arma::vec &x, arma::vec &y, arma::vec &z);
  // Convert from x-y coordinates in equilateral triangle to r-s coordinates of
  // the reference triagnle
  void xy2rs(const arma::vec &x, const arma::vec &y, arma::vec &r,
             arma::vec &s);
  // Convert from x-y-z coordinates to r-s-t coordinates
  void xyz2rst(const arma::vec &x, const arma::vec &y, const arma::vec &z,
               arma::vec &r, arma::vec &s, arma::vec &t);
  // Convert from r-s coordinates to a-b coordinates
  void rs2ab(const arma::vec &r, const arma::vec &s, arma::vec &a,
             arma::vec &b);
  // Convert from r-s-t coordinates to a-b-c coordinates
  void rst2abc(const arma::vec &r, const arma::vec &s, const arma::vec &t,
               arma::vec &a, arma::vec &b, arma::vec &c);

  /*********************************
  * Misc
  **********************************/

  // Calculate number of points, number of face points from order N
  void numNodes2D(const int N, int *Np, int *Nfp);
  void numNodes3D(const int N, int *Np, int *Nfp);
  // Convert from global x-y coords to r-s coords
  void global_xy_to_rs(const double x, const double y, double &r, double &s,
                       const double *cellX, const double *cellY);
  // Convert from r-s coords to global x-y coords
  void rs_to_global_xy(const double r, const double s, double &x, double &y,
                       const double *cellX, const double *cellY);
  
  /*********************************
  * Interpolating values within a cell
  **********************************/

  double val_at_pt_3d(const double r, const double s, const double t,
                      const double *modal, const int N);
  void grad_at_pt_3d(const double r, const double s, const double t,
                     const double *modal, const int N, double &dr,
                     double &ds, double &dt);
  double val_at_pt_N_1_3d(const double r, const double s, const double t,
                          const double *modal, const int N);
}

#endif
