#ifndef __DG_UTILS_H
#define __DG_UTILS_H

#include "dg_compiler_defs.h"

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
  arma::mat eMat2D(const arma::vec &r, const arma::vec &s,
                   const arma::uvec &fmask, const int N);
  arma::mat lift2D(const arma::vec &r, const arma::vec &s,
                   const arma::uvec &fmask, const arma::mat &V, const int N);
  // Calculate 3D surface to volume lift operator without inv Mass
  arma::mat eMat3D(const arma::vec &r, const arma::vec &s, const arma::vec &t,
                   const arma::uvec &fmask, const int N);
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
  void faceMassMatrix2D(const arma::vec &r, const arma::vec &s,
                        const arma::uvec &fmask, const arma::mat &v,
                        const int N, arma::mat &face0, arma::mat &face1,
                        arma::mat &face2);
  void faceMassMatrix3D(const arma::vec &r, const arma::vec &s,
                        const arma::vec &t, const arma::uvec &fmask,
                        const arma::mat &v, const int N, arma::mat &face0,
                        arma::mat &face1, arma::mat &face2, arma::mat &face3);
  
  arma::mat cubaturePMat2D(const arma::vec &r, const arma::vec &s,
                           const arma::vec &cubr, const arma::vec &cubs,
                           const int N);
  
  void cubaturePDwMat2D(const arma::vec &r, const arma::vec &s,
                        const arma::vec &cubr, const arma::vec &cubs,
                        const int N, arma::mat &cubDrw, arma::mat &cubDsw);

  arma::mat cubaturePMat3D(const arma::vec &r, const arma::vec &s, const arma::vec &t, 
                           const arma::vec &cubr, const arma::vec &cubs,
                           const arma::vec &cubt, const int N);
  
  void cubaturePDwMat3D(const arma::vec &r, const arma::vec &s, const arma::vec &t, 
                        const arma::vec &cubr, const arma::vec &cubs, const arma::vec &cubt, 
                        const int N, arma::mat &cubDrw, arma::mat &cubDsw, arma::mat &cubDtw);

  void cubatureSurface3d(const arma::vec &r, const arma::vec &s, const arma::vec &t, 
                         const arma::uvec &fmask, const arma::vec &cubr, const arma::vec &cubs,
                         const arma::vec &cubw, const int N, arma::mat &interp, arma::mat &lift);

  // Calculate a filter matrix that targets modes of degree Nc and above
  arma::mat filterMatrix3D(const arma::mat &v, const arma::mat &invV,
                           const int N, const int Nc, const int s,
                           const DG_FP alpha);

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
  void global_xy_to_rs(const DG_FP x, const DG_FP y, DG_FP &r, DG_FP &s,
                       const DG_FP *cellX, const DG_FP *cellY);
  // Convert from r-s coords to global x-y coords
  void rs_to_global_xy(const DG_FP r, const DG_FP s, DG_FP &x, DG_FP &y,
                       const DG_FP *cellX, const DG_FP *cellY);

  /*********************************
  * Interpolating values within a cell
  **********************************/

  DG_FP val_at_pt_2d(const DG_FP r, const DG_FP s, const DG_FP *modal, 
                     const int N);
  void grad_at_pt_2d(const DG_FP r, const DG_FP s, const DG_FP *modal, 
                     const int N, DG_FP &dr, DG_FP &ds);
  DG_FP val_at_pt_3d(const DG_FP r, const DG_FP s, const DG_FP t,
                      const DG_FP *modal, const int N);
  void grad_at_pt_3d(const DG_FP r, const DG_FP s, const DG_FP t,
                     const DG_FP *modal, const int N, DG_FP &dr,
                     DG_FP &ds, DG_FP &dt);
  DG_FP val_at_pt_N_1_3d(const DG_FP r, const DG_FP s, const DG_FP t,
                          const DG_FP *modal, const int N);
  std::vector<DG_FP> val_at_pt_N_1_3d_get_simplexes(const std::vector<DG_FP> &r,
                        const std::vector<DG_FP> &s, const std::vector<DG_FP> &t,
                        const int N);
  std::vector<DG_FP> val_at_pt_N_1_2d_get_simplexes(const std::vector<DG_FP> &r,
                        const std::vector<DG_FP> &s, const int N);
  
  template<std::size_t DIM>
  class Vec {
  private:
    double vals[DIM];
  public:
    Vec() {
      for(int i = 0; i < DIM; i++)
        vals[i] = 0.0;
    }
    template<typename... Args>
    explicit Vec(Args... args): vals{args...} {}

    double sqr_magnitude() {
      double result = 0.0;
      for(int i = 0; i < DIM; i++) {
        result += vals[i] * vals[i];
      }
      return result;
    }

    double magnitude() {
      return sqrt(sqr_magnitude());
    }

    double dot(const Vec<DIM> &rhs) {
      double result = 0.0;
      for(int i = 0; i < DIM; i++) {
        result += vals[i] * rhs[i];
      }
      return result;
    }

    double& operator[](std::size_t idx) { return vals[idx]; }
    const double& operator[](std::size_t idx) const { return vals[idx]; }
    Vec<DIM>& operator+=(const Vec<DIM> &rhs) {
      for(int i = 0; i < DIM; i++) {
        vals[i] += rhs[i];
      }
      return *this;
    }
    Vec<DIM>& operator-=(const Vec<DIM> &rhs) {
      for(int i = 0; i < DIM; i++) {
        vals[i] -= rhs[i];
      }
      return *this;
    }

    friend bool operator==(const Vec<DIM> &lhs, const Vec<DIM> &rhs) {
      for(int i = 0; i < DIM; i++) {
        if(fabs(lhs[i] - rhs[i]) > 1e-8)
          return false;
      }
      return true;
    }
    friend bool operator!=(const Vec<DIM> &lhs, const Vec<DIM> &rhs) { 
      return !(lhs == rhs); 
    }
    friend bool operator<(const Vec<DIM> &lhs, const Vec<DIM> &rhs) {
      for(int i = 0; i < DIM; i++) {
        if(fabs(lhs[i] - rhs[i]) >= 1e-8)
          return lhs[i] < rhs[i];
      }
      return false;
    }
    friend bool operator>(const Vec<DIM>& lhs, const Vec<DIM>& rhs) { return rhs < lhs; }
    friend bool operator<=(const Vec<DIM>& lhs, const Vec<DIM>& rhs) { return !(lhs > rhs); }
    friend bool operator>=(const Vec<DIM>& lhs, const Vec<DIM>& rhs) { return !(lhs < rhs); }
    friend Vec<DIM> operator+(Vec<DIM> lhs, const Vec<DIM> &rhs) {
      lhs += rhs;
      return lhs; 
    }
    friend Vec<DIM> operator-(Vec<DIM> lhs, const Vec<DIM> &rhs) {
      lhs -= rhs;
      return lhs; 
    }
    friend Vec<DIM> operator*(const double &lhs, const Vec<DIM> &rhs) {
      Vec<DIM> result = rhs;
      for(int i = 0; i < DIM; i++) {
        result[i] *= lhs;
      }
      return result; 
    }
    friend Vec<DIM> operator*(const Vec<DIM> &lhs, const double &rhs) { return rhs * lhs; }
  };
}

#endif
