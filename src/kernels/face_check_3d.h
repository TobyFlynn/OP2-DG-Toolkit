inline void face_check_3d(const int *order, const int *faceNum,
                          const DG_FP **x, const DG_FP **y, 
                          const DG_FP **z, int *fmaskL_corrected, 
                          int *fmaskR_corrected, int *num) {
  const int p = *order;
  const int dg_npf = DG_CONSTANTS_TK[(p - 1) * DG_NUM_CONSTANTS + 1];
  const int faceNumL = faceNum[0];
  const int faceNumR = faceNum[1];
  const int *fmaskL = &FMASK_TK[(p - 1) * DG_NUM_FACES * DG_NPF + faceNumL * dg_npf];
  const int *fmaskR = &FMASK_TK[(p - 1) * DG_NUM_FACES * DG_NPF + faceNumR * dg_npf];

  for(int i = 0; i < dg_npf; i++) {
    fmaskR_corrected[i] = 0;
    fmaskL_corrected[i] = 0;
    for(int j = 0; j < dg_npf; j++) {
      const int fmaskL_ind_i = fmaskL[i];
      const int fmaskL_ind_j = fmaskL[j];
      const int fmaskR_ind_i = fmaskR[i];
      const int fmaskR_ind_j = fmaskR[j];
      if(fp_equal(x[0][fmaskL_ind_i], x[1][fmaskR_ind_j]) &&
         fp_equal(y[0][fmaskL_ind_i], y[1][fmaskR_ind_j]) &&
         fp_equal(z[0][fmaskL_ind_i], z[1][fmaskR_ind_j]))
        fmaskR_corrected[i] = fmaskR[j];
      if(fp_equal(x[1][fmaskR_ind_i], x[0][fmaskL_ind_j]) &&
         fp_equal(y[1][fmaskR_ind_i], y[0][fmaskL_ind_j]) &&
         fp_equal(z[1][fmaskR_ind_i], z[0][fmaskL_ind_j]))
        fmaskL_corrected[i] = fmaskL[j];
    }
  }

  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL[i];
    const int fmaskR_ind = fmaskR_corrected[i];
    if(!fp_equal(x[0][fmaskL_ind], x[1][fmaskR_ind]) ||
       !fp_equal(y[0][fmaskL_ind], y[1][fmaskR_ind]) ||
       !fp_equal(z[0][fmaskL_ind], z[1][fmaskR_ind]))
      (*num)++;
  }

  for(int i = 0; i < dg_npf; i++) {
    const int fmaskL_ind = fmaskL_corrected[i];
    const int fmaskR_ind = fmaskR[i];
    if(!fp_equal(x[0][fmaskL_ind], x[1][fmaskR_ind]) ||
       !fp_equal(y[0][fmaskL_ind], y[1][fmaskR_ind]) ||
       !fp_equal(z[0][fmaskL_ind], z[1][fmaskR_ind]))
      (*num)++;
  }
}