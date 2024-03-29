#include "dg_utils.h"

void DGUtils::numNodes2D(const int N, int *Np, int *Nfp) {
  // Number of points per face of triangluar element
  *Nfp = N + 1;
  // Number of pointer per element
  *Np = (N + 1) * (N + 2) / 2;
}

void DGUtils::numNodes3D(const int N, int *Np, int *Nfp) {
  // Number of points per face of the element
  *Nfp = (N + 1) * (N + 2) / 2;
  // Number of pointer per element
  *Np = (N + 1) * (N + 2) * (N + 3) / 6;
}

// Convert from global x-y coords to r-s coords
void DGUtils::global_xy_to_rs(const DG_FP x, const DG_FP y, DG_FP &r,
                              DG_FP &s, const DG_FP *cellX,
                              const DG_FP *cellY) {
  DG_FP l2 = (cellY[1] - cellY[2]) * (x - cellX[2]) + (cellX[2] - cellX[1]) * (y - cellY[2]);
  l2 = l2 / ((cellY[1] - cellY[2]) * (cellX[0] - cellX[2]) + (cellX[2] - cellX[1]) * (cellY[0] - cellY[2]));
  DG_FP l3 = (cellY[2] - cellY[0]) * (x - cellX[2]) + (cellX[0] - cellX[2]) * (y - cellY[2]);
  l3 = l3 / ((cellY[1] - cellY[2]) * (cellX[0] - cellX[2]) + (cellX[2] - cellX[1]) * (cellY[0] - cellY[2]));
  DG_FP l1 = 1.0 - l2 - l3;
  s = 2.0 * l1 - 1.0;
  r = 2.0 * l3 - 1.0;
}

// Convert from r-s coords to global x-y coords
void DGUtils::rs_to_global_xy(const DG_FP r, const DG_FP s, DG_FP &x,
                              DG_FP &y, const DG_FP *cellX,
                              const DG_FP *cellY) {
  x = 0.5 * (-(r + s) * cellX[0] + (1.0 + r) * cellX[1] + (1.0 + s) * cellX[2]);
  y = 0.5 * (-(r + s) * cellY[0] + (1.0 + r) * cellY[1] + (1.0 + s) * cellY[2]);
}
