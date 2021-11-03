#ifndef __DG_UTILS_H
#define __DG_UTILS_H

#include <vector>

namespace DGUtils {

  const double PI = 3.141592653589793238463;

  /*********************************
  * Calculating polynomials
  **********************************/

  // Calc Nth order Gauss quadature points and weights of Jacobi polynomial
  void jacobiGQ(const double alpha, const double beta, const int n,
                std::vector<double> &x, std::vector<double> &w);
  // Calc Nth order Gauss Lobatto quadature points of Jacobi polynomial
  std::vector<double> jacobiGL(const double alpha, const double beta,
                               const int N);

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

  // Calculates number of points, number of face points from order N
  void basic_constants(const int N, int *Np, int *Nfp);
}

#endif
