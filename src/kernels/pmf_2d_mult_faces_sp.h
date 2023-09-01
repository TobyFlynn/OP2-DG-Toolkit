inline void pmf_2d_mult_faces_sp(const int *order, const int *faceNum,
                            const bool *reverse, const DG_FP *nx,
                            const DG_FP *ny, const DG_FP *fscale,
                            const DG_FP *sJ, const float **in,
                            const float **in_x, const float **in_y,
                            float **l_x, float **l_y, float **out) {
  const int dg_np  = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS];
  const int dg_npf = DG_CONSTANTS_TK[(*order - 1) * DG_NUM_CONSTANTS + 1];

  const int findR = faceNum[1] * dg_npf;
  const int *fmask  = &FMASK_TK[(*order - 1) * DG_NUM_FACES * DG_NPF];
  const int *fmaskL = &fmask[faceNum[0] * dg_npf];
  const int *fmaskR = &fmask[faceNum[1] * dg_npf];

  const float gtau = (float)(2.0 * (*order + 1) * (*order + 2) * fmax(fscale[0], fscale[1]));

  const bool rev = *reverse;
  const float nxL = (float)nx[0];
  const float nyL = (float)ny[0];
  const float nxR = (float)nx[1];
  const float nyR = (float)ny[1];
  const float sJL = (float)sJ[0];
  const float sJR = (float)sJ[1];
  for(int j = 0; j < dg_npf; j++) {
    const int fmaskL_ind = fmaskL[j];
    const int fmaskR_ind = rev ? fmaskR[dg_npf - j - 1] : fmaskR[j];
    const float diffL_u = in[0][fmaskL_ind] - in[1][fmaskR_ind];
    const float diff_u_x = in_x[1][fmaskR_ind] + in_x[0][fmaskL_ind];
    const float diff_u_y = in_y[1][fmaskR_ind] + in_y[0][fmaskL_ind];
    const float diffL_u_grad = nxL * diff_u_x + nyL * diff_u_y;
    const float diffR_u_grad = nxR * diff_u_x + nyR * diff_u_y;

    const int indL = faceNum[0] * dg_npf + j;
    out[0][indL] = 0.5 * sJL * (gtau * diffL_u - diffL_u_grad);
    const float l_tmpL = 0.5 * sJL * -diffL_u;
    l_x[0][indL] = nxL * l_tmpL;
    l_y[0][indL] = nyL * l_tmpL;

    const int indR = rev ? faceNum[1] * dg_npf + dg_npf - j - 1 : faceNum[1] * dg_npf + j;
    out[1][indR] = 0.5 * sJR * (gtau * -diffL_u - diffR_u_grad);
    const float l_tmpR = 0.5 * sJR * diffL_u;
    l_x[1][indR] = nxR * l_tmpR;
    l_y[1][indR] = nyR * l_tmpR;
  }
}
