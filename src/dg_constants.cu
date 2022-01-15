#include "dg_constants.h"

#include "dg_global_constants.h"

DGConstants::DGConstants(const int n) {
  setup(n);

  // Cubature constants
  cudaMalloc((void**)&cubDr_d, DG_CUB_NP * DG_NP * sizeof(double));
  cudaMemcpy(cubDr_d, &cubDr_g[(N - 1) * DG_CUB_NP * DG_NP], DG_CUB_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&cubDs_d, DG_CUB_NP * DG_NP * sizeof(double));
  cudaMemcpy(cubDs_d, &cubDs_g[(N - 1) * DG_CUB_NP * DG_NP], DG_CUB_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&cubV_d, DG_CUB_NP * DG_NP * sizeof(double));
  cudaMemcpy(cubV_d, &cubV_g[(N - 1) * DG_CUB_NP * DG_NP], DG_CUB_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&cubVDr_d, DG_CUB_NP * DG_NP * sizeof(double));
  cudaMemcpy(cubVDr_d, &cubVDr_g[(N - 1) * DG_CUB_NP * DG_NP], DG_CUB_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&cubVDs_d, DG_CUB_NP * DG_NP * sizeof(double));
  cudaMemcpy(cubVDs_d, &cubVDs_g[(N - 1) * DG_CUB_NP * DG_NP], DG_CUB_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&cubW_d, DG_CUB_NP * sizeof(double));
  cudaMemcpy(cubW_d, &cubW_g[(N - 1) * DG_CUB_NP], DG_CUB_NP * sizeof(double), cudaMemcpyHostToDevice);
  // Grad constants
  cudaMalloc((void**)&Dr_d, DG_NP * DG_NP * sizeof(double));
  cudaMemcpy(Dr_d, &Dr_g[(N - 1) * DG_NP * DG_NP], DG_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&Drw_d, DG_NP * DG_NP * sizeof(double));
  cudaMemcpy(Drw_d, &Drw_g[(N - 1) * DG_NP * DG_NP], DG_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&Ds_d, DG_NP * DG_NP * sizeof(double));
  cudaMemcpy(Ds_d, &Ds_g[(N - 1) * DG_NP * DG_NP], DG_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&Dsw_d, DG_NP * DG_NP * sizeof(double));
  cudaMemcpy(Dsw_d, &Dsw_g[(N - 1) * DG_NP * DG_NP], DG_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  // Gauss constants
  cudaMalloc((void**)&gaussW_d, DG_GF_NP * sizeof(double));
  cudaMemcpy(gaussW_d, &gaussW_g[(N - 1) * DG_GF_NP], DG_GF_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&gF0Dr_d, DG_GF_NP * DG_NP * sizeof(double));
  cudaMemcpy(gF0Dr_d, &gF0Dr_g[(N - 1) * DG_GF_NP * DG_NP], DG_GF_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&gF0Ds_d, DG_GF_NP * DG_NP * sizeof(double));
  cudaMemcpy(gF0Ds_d, &gF0Ds_g[(N - 1) * DG_GF_NP * DG_NP], DG_GF_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&gF1Dr_d, DG_GF_NP * DG_NP * sizeof(double));
  cudaMemcpy(gF1Dr_d, &gF1Dr_g[(N - 1) * DG_GF_NP * DG_NP], DG_GF_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&gF1Ds_d, DG_GF_NP * DG_NP * sizeof(double));
  cudaMemcpy(gF1Ds_d, &gF1Ds_g[(N - 1) * DG_GF_NP * DG_NP], DG_GF_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&gF2Dr_d, DG_GF_NP * DG_NP * sizeof(double));
  cudaMemcpy(gF2Dr_d, &gF2Dr_g[(N - 1) * DG_GF_NP * DG_NP], DG_GF_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&gF2Ds_d, DG_GF_NP * DG_NP * sizeof(double));
  cudaMemcpy(gF2Ds_d, &gF2Ds_g[(N - 1) * DG_GF_NP * DG_NP], DG_GF_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&gFInterp0_d, DG_GF_NP * DG_NP * sizeof(double));
  cudaMemcpy(gFInterp0_d, &gFInterp0_g[(N - 1) * DG_GF_NP * DG_NP], DG_GF_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&gFInterp1_d, DG_GF_NP * DG_NP * sizeof(double));
  cudaMemcpy(gFInterp1_d, &gFInterp1_g[(N - 1) * DG_GF_NP * DG_NP], DG_GF_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&gFInterp2_d, DG_GF_NP * DG_NP * sizeof(double));
  cudaMemcpy(gFInterp2_d, &gFInterp2_g[(N - 1) * DG_GF_NP * DG_NP], DG_GF_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&gInterp_d, DG_G_NP * DG_NP * sizeof(double));
  cudaMemcpy(gInterp_d, &gInterp_g[(N - 1) * DG_G_NP * DG_NP], DG_G_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  // Other constants
  cudaMalloc((void**)&invMass_d, DG_NP * DG_NP * sizeof(double));
  cudaMemcpy(invMass_d, &invMass_g[(N - 1) * DG_NP * DG_NP], DG_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&lift_d, DG_NP * DG_NPF * 3 * sizeof(double));
  cudaMemcpy(lift_d, &lift_g[(N - 1) * DG_NP * 3 * DG_NPF], DG_NP * DG_NPF * 3 * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&mass_d, DG_NP * DG_NP * sizeof(double));
  cudaMemcpy(mass_d, &mass_g[(N - 1) * DG_NP * DG_NP], DG_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&v_d, DG_NP * DG_NP * sizeof(double));
  cudaMemcpy(v_d, &v_g[(N - 1) * DG_NP * DG_NP], DG_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&invV_d, DG_NP * DG_NP * sizeof(double));
  cudaMemcpy(invV_d, &invV_g[(N - 1) * DG_NP * DG_NP], DG_NP * DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&r_d, DG_NP * sizeof(double));
  cudaMemcpy(r_d, &r_g[(N - 1) * DG_NP], DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&s_d, DG_NP * sizeof(double));
  cudaMemcpy(s_d, &s_g[(N - 1) * DG_NP], DG_NP * sizeof(double), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&ones_d, DG_NP * sizeof(double));
  cudaMemcpy(ones_d, &ones_g[(N - 1) * DG_NP], DG_NP * sizeof(double), cudaMemcpyHostToDevice);
}

DGConstants::~DGConstants() {
  // Cubature constants
  cudaFree(cubDr_d);
  cudaFree(cubDs_d);
  cudaFree(cubV_d);
  cudaFree(cubVDr_d);
  cudaFree(cubVDs_d);
  cudaFree(cubW_d);
  // Grad constants
  cudaFree(Dr_d);
  cudaFree(Drw_d);
  cudaFree(Ds_d);
  cudaFree(Dsw_d);
  // Gauss constants
  cudaFree(gaussW_d);
  cudaFree(gF0Dr_d);
  cudaFree(gF0Ds_d);
  cudaFree(gF1Dr_d);
  cudaFree(gF1Ds_d);
  cudaFree(gF2Dr_d);
  cudaFree(gF2Ds_d);
  cudaFree(gFInterp0_d);
  cudaFree(gFInterp1_d);
  cudaFree(gFInterp2_d);
  cudaFree(gInterp_d);
  // Other constants
  cudaFree(invMass_d);
  cudaFree(lift_d);
  cudaFree(mass_d);
  cudaFree(v_d);
  cudaFree(invV_d);
  cudaFree(r_d);
  cudaFree(s_d);
  cudaFree(ones_d);
}

double* DGConstants::get_ptr(Constant_Matrix mat) {
  switch(mat) {
    case CUB_DR:
      return cubDr_d;
    case CUB_DS:
      return cubDs_d;
    case CUB_V:
      return cubV_d;
    case CUB_VDR:
      return cubVDr_d;
    case CUB_VDS:
      return cubVDs_d;
    case CUB_W:
      return cubW_d;
    case DR:
      return Dr_d;
    case DRW:
      return Drw_d;
    case DS:
      return Ds_d;
    case DSW:
      return Dsw_d;
    case GAUSS_W:
      return gaussW_d;
    case GAUSS_F0DR:
      return gF0Dr_d;
    case GAUSS_F0DS:
      return gF0Ds_d;
    case GAUSS_F1DR:
      return gF1Dr_d;
    case GAUSS_F1DS:
      return gF1Ds_d;
    case GAUSS_F2DR:
      return gF2Dr_d;
    case GAUSS_F2DS:
      return gF2Ds_d;
    case GAUSS_FINTERP0:
      return gFInterp0_d;
    case GAUSS_FINTERP1:
      return gFInterp1_d;
    case GAUSS_FINTERP2:
      return gFInterp2_d;
    case GAUSS_INTERP:
      return gInterp_d;
    case INV_MASS:
      return invMass_d;
    case LIFT:
      return lift_d;
    case MASS:
      return mass_d;
    case V:
      return v_d;
    case INV_V:
      return invV_d;
    case R:
      return r_d;
    case S:
      return s_d;
    case ONES:
      return ones_d;
  }
  return nullptr;
}
