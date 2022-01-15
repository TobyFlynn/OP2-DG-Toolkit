#include "dg_utils.h"

void DGUtils::basic_constants(const int N, int *Np, int *Nfp) {
  // Number of points per face of triangluar element
  *Nfp = N + 1;
  // Number of pointer per element
  *Np = (N + 1) * (N + 2) / 2;
}
