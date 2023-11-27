inline void pmf_3d_mult_indirect_sp(const int * __restrict__ order, const int * __restrict__ faceNum,
                      const int * __restrict__ fmaskL_corrected, const int * __restrict__ fmaskR_corrected,
                      const DG_FP * __restrict__ nx, const DG_FP * __restrict__ ny, const DG_FP * __restrict__ nz,
                      const DG_FP * __restrict__ sJ, const DG_FP * __restrict__ fscale, const float ** __restrict__ in,
                      const float ** __restrict__ in_x, const float ** __restrict__ in_y,
                      const float ** __restrict__ in_z, float ** __restrict__ pen_out, float ** __restrict__ out_x,
                      float ** __restrict__ out_y, float ** __restrict__ out_z) {
  const int p = *order;
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];
  const int *fmask = &FMASK_TK[(p - 1) * DG_NUM_FACES * DG_NPF];
  const int faceNumL = faceNum[0];
  const int faceNumR = faceNum[1];
  const int *fmaskL = &fmask[faceNumL * dg_npf];
  const int *fmaskR = &fmask[faceNumR * dg_npf];

  const float _tau = 2.0f * (p + 1) * (p + 2) * fmax(fscale[0], fscale[1]);

  const float _nxL = (float)nx[0];
  const float _nyL = (float)ny[0];
  const float _nzL = (float)nz[0];
  const float _sJL = (float)sJ[0];
  float _half_jump[DG_NPF], _avg_x[DG_NPF], _avg_y[DG_NPF], _avg_z[DG_NPF];
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL[i];
    const int fmaskR_ind = fmaskR_corrected[i];
    _half_jump[i] = 0.5f * (in[0][fmaskL_ind] - in[1][fmaskR_ind]);
  }
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL[i];
    const int fmaskR_ind = fmaskR_corrected[i];
    _avg_x[i] = 0.5f * (in_x[0][fmaskL_ind] + in_x[1][fmaskR_ind]);
  }
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL[i];
    const int fmaskR_ind = fmaskR_corrected[i];
    _avg_y[i] = 0.5f * (in_y[0][fmaskL_ind] + in_y[1][fmaskR_ind]);
  }
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL[i];
    const int fmaskR_ind = fmaskR_corrected[i];
    _avg_z[i] = 0.5f * (in_z[0][fmaskL_ind] + in_z[1][fmaskR_ind]);
  }
  for(int i = 0; i < dg_npf; i++) {
    const float _sum = _nxL * _avg_x[i] + _nyL * _avg_y[i] + _nzL * _avg_z[i];
    pen_out[0][faceNumL * dg_npf + i] = _sJL * (_tau * _half_jump[i] - _sum);
    out_x[0][faceNumL * dg_npf + i]   = _nxL * _sJL * -_half_jump[i];
    out_y[0][faceNumL * dg_npf + i]   = _nyL * _sJL * -_half_jump[i];
    out_z[0][faceNumL * dg_npf + i]   = _nzL * _sJL * -_half_jump[i];
  }
  /*
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL[i];
    const int fmaskR_ind = fmaskR_corrected[i];

    const float _half_jump = 0.5f * (in[0][fmaskL_ind] - in[1][fmaskR_ind]);
    const float _avg_x = 0.5f * (in_x[0][fmaskL_ind] + in_x[1][fmaskR_ind]);
    const float _avg_y = 0.5f * (in_y[0][fmaskL_ind] + in_y[1][fmaskR_ind]);
    const float _avg_z = 0.5f * (in_z[0][fmaskL_ind] + in_z[1][fmaskR_ind]);
    const float _sum   = _nxL * _avg_x + _nyL * _avg_y + _nzL * _avg_z;

    pen_out[0][faceNumL * dg_npf + i] = _sJL * (_tau * _half_jump - _sum);
    out_x[0][faceNumL * dg_npf + i]   = _nxL * _sJL * -_half_jump;
    out_y[0][faceNumL * dg_npf + i]   = _nyL * _sJL * -_half_jump;
    out_z[0][faceNumL * dg_npf + i]   = _nzL * _sJL * -_half_jump;
  }
  */
  const float _nxR = (float)nx[1];
  const float _nyR = (float)ny[1];
  const float _nzR = (float)nz[1];
  const float _sJR = (float)sJ[1];
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL_corrected[i];
    const int fmaskR_ind = fmaskR[i];
    _half_jump[i] = 0.5f * (in[1][fmaskR_ind] - in[0][fmaskL_ind]);
  }
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL_corrected[i];
    const int fmaskR_ind = fmaskR[i];
    _avg_x[i] = 0.5f * (in_x[0][fmaskL_ind] + in_x[1][fmaskR_ind]);
  }
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL_corrected[i];
    const int fmaskR_ind = fmaskR[i];
    _avg_y[i] = 0.5f * (in_y[0][fmaskL_ind] + in_y[1][fmaskR_ind]);
  }
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL_corrected[i];
    const int fmaskR_ind = fmaskR[i];
    _avg_z[i] = 0.5f * (in_z[0][fmaskL_ind] + in_z[1][fmaskR_ind]);
  }
  for(int i = 0; i < dg_npf; i++) {
    const float _sum = _nxR * _avg_x[i] + _nyR * _avg_y[i] + _nzR * _avg_z[i];
    pen_out[1][faceNumR * dg_npf + i] = _sJR * (_tau * _half_jump[i] - _sum);
    out_x[1][faceNumR * dg_npf + i]   = _nxR * _sJR * -_half_jump[i];
    out_y[1][faceNumR * dg_npf + i]   = _nyR * _sJR * -_half_jump[i];
    out_z[1][faceNumR * dg_npf + i]   = _nzR * _sJR * -_half_jump[i];
  }
  /*
  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL_corrected[i];
    const int fmaskR_ind = fmaskR[i];

    const float _half_jump = 0.5f * (in[1][fmaskR_ind] - in[0][fmaskL_ind]);
    const float _avg_x = 0.5f * (in_x[0][fmaskL_ind] + in_x[1][fmaskR_ind]);
    const float _avg_y = 0.5f * (in_y[0][fmaskL_ind] + in_y[1][fmaskR_ind]);
    const float _avg_z = 0.5f * (in_z[0][fmaskL_ind] + in_z[1][fmaskR_ind]);
    const float _sum   = _nxR * _avg_x + _nyR * _avg_y + _nzR * _avg_z;

    pen_out[1][faceNumR * dg_npf + i] = _sJR * (_tau * _half_jump - _sum);
    out_x[1][faceNumR * dg_npf + i]   = _nxR * _sJR * -_half_jump;
    out_y[1][faceNumR * dg_npf + i]   = _nyR * _sJR * -_half_jump;
    out_z[1][faceNumR * dg_npf + i]   = _nzR * _sJR * -_half_jump;
  }
  */
}
