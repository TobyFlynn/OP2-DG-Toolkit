inline void pmf_3d_mult_indirect(const int * __restrict__ order, const int * __restrict__ faceNum,
                      const int * __restrict__ fmaskL_corrected, const int * __restrict__ fmaskR_corrected,
                      const DG_FP * __restrict__ nx, const DG_FP * __restrict__ ny, const DG_FP * __restrict__ nz,
                      const DG_FP * __restrict__ sJ, const DG_FP * __restrict__ fscale, const DG_FP ** __restrict__ in,
                      const DG_FP ** __restrict__ in_x, const DG_FP ** __restrict__ in_y,
                      const DG_FP ** __restrict__ in_z, DG_FP ** __restrict__ pen_out, DG_FP ** __restrict__ out_x,
                      DG_FP ** __restrict__ out_y, DG_FP ** __restrict__ out_z) {
  const int p = *order;
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];
  const int *fmask = &FMASK_TK[(p - 1) * DG_NUM_FACES * DG_NPF];
  const int faceNumL = faceNum[0];
  const int faceNumR = faceNum[1];
  const int *fmaskL = &fmask[faceNumL * dg_npf];
  const int *fmaskR = &fmask[faceNumR * dg_npf];

  const DG_FP _tau = 2.0 * (p + 1) * (p + 2) * fmax(fscale[0], fscale[1]);

  const DG_FP _nxL = nx[0];
  const DG_FP _nyL = ny[0];
  const DG_FP _nzL = nz[0];
  const DG_FP _sJL = sJ[0];
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL[i];
    const int fmaskR_ind = fmaskR_corrected[i];

    const DG_FP _minus_half_jump = 0.5 * (in[1][fmaskR_ind] - in[0][fmaskL_ind]);
    const DG_FP _avg_x = in_x[0][fmaskL_ind] + in_x[1][fmaskR_ind];
    const DG_FP _avg_y = in_y[0][fmaskL_ind] + in_y[1][fmaskR_ind];
    const DG_FP _avg_z = in_z[0][fmaskL_ind] + in_z[1][fmaskR_ind];
    const DG_FP _sum   = 0.5 * (_nxL * _avg_x + _nyL * _avg_y + _nzL * _avg_z);

    pen_out[0][faceNumL * dg_npf + i] = _sJL * (_tau * -_minus_half_jump - _sum);
    out_x[0][faceNumL * dg_npf + i]   = _nxL * _sJL * _minus_half_jump;
    out_y[0][faceNumL * dg_npf + i]   = _nyL * _sJL * _minus_half_jump;
    out_z[0][faceNumL * dg_npf + i]   = _nzL * _sJL * _minus_half_jump;
  }
  
  const DG_FP _nxR = nx[1];
  const DG_FP _nyR = ny[1];
  const DG_FP _nzR = nz[1];
  const DG_FP _sJR = sJ[1];
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL_corrected[i];
    const int fmaskR_ind = fmaskR[i];

    const DG_FP _minus_half_jump = 0.5 * (in[0][fmaskL_ind] - in[1][fmaskR_ind]);
    const DG_FP _avg_x = in_x[0][fmaskL_ind] + in_x[1][fmaskR_ind];
    const DG_FP _avg_y = in_y[0][fmaskL_ind] + in_y[1][fmaskR_ind];
    const DG_FP _avg_z = in_z[0][fmaskL_ind] + in_z[1][fmaskR_ind];
    const DG_FP _sum   = 0.5 * (_nxR * _avg_x + _nyR * _avg_y + _nzR * _avg_z);

    pen_out[1][faceNumR * dg_npf + i] = _sJR * (_tau * -_minus_half_jump - _sum);
    out_x[1][faceNumR * dg_npf + i]   = _nxR * _sJR * _minus_half_jump;
    out_y[1][faceNumR * dg_npf + i]   = _nyR * _sJR * _minus_half_jump;
    out_z[1][faceNumR * dg_npf + i]   = _nzR * _sJR * _minus_half_jump;
  }
}
