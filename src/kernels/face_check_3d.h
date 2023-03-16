inline void face_check_3d(const int *order, const int *faceNum,
                          const int *periodicFace, const DG_FP **x,
                          const DG_FP **y, const DG_FP **z,
                          int *fmaskL_corrected, int * fmaskR_corrected,
                          int *num) {
  const int dg_npf = DG_CONSTANTS_TK[(*order - 1) * 2 + 1];
  const int *fmask  = &FMASK_TK[(*order - 1) * 4 * DG_NPF];
  const int *fmaskL = &fmask[faceNum[0] * dg_npf];
  const int *fmaskR = &fmask[faceNum[1] * dg_npf];

  if(*periodicFace == 0) {
    for(int i = 0; i < dg_npf; i++) {
      fmaskR_corrected[i] = 0;
      fmaskL_corrected[i] = 0;
      for(int j = 0; j < dg_npf; j++) {
        if(fp_equal(x[0][fmaskL[i]], x[1][fmaskR[j]]) &&
           fp_equal(y[0][fmaskL[i]], y[1][fmaskR[j]]) &&
           fp_equal(z[0][fmaskL[i]], z[1][fmaskR[j]]))
          fmaskR_corrected[i] = fmaskR[j];
        if(fp_equal(x[1][fmaskR[i]], x[0][fmaskL[j]]) &&
           fp_equal(y[1][fmaskR[i]], y[0][fmaskL[j]]) &&
           fp_equal(z[1][fmaskR[i]], z[0][fmaskL[j]]))
          fmaskL_corrected[i] = fmaskL[j];
      }
    }

    for(int i = 0; i < dg_npf; i++) {
      if(!fp_equal(x[0][fmaskL[i]], x[1][fmaskR_corrected[i]]) ||
         !fp_equal(y[0][fmaskL[i]], y[1][fmaskR_corrected[i]]) ||
         !fp_equal(z[0][fmaskL[i]], z[1][fmaskR_corrected[i]]))
        (*num)++;
    }
  } else if(*periodicFace == 1) {
    for(int i = 0; i < dg_npf; i++) {
      fmaskR_corrected[i] = 0;
      fmaskL_corrected[i] = 0;
      for(int j = 0; j < dg_npf; j++) {
        if(fp_equal(y[0][fmaskL[i]], y[1][fmaskR[j]]) &&
           fp_equal(z[0][fmaskL[i]], z[1][fmaskR[j]]))
          fmaskR_corrected[i] = fmaskR[j];
        if(fp_equal(y[1][fmaskR[i]], y[0][fmaskL[j]]) &&
           fp_equal(z[1][fmaskR[i]], z[0][fmaskL[j]]))
          fmaskL_corrected[i] = fmaskL[j];
      }
    }

    for(int i = 0; i < dg_npf; i++) {
      if(!fp_equal(y[0][fmaskL[i]], y[1][fmaskR_corrected[i]]) ||
         !fp_equal(z[0][fmaskL[i]], z[1][fmaskR_corrected[i]]))
        (*num)++;
    }
  } else if(*periodicFace == 2) {
    for(int i = 0; i < dg_npf; i++) {
      fmaskR_corrected[i] = 0;
      fmaskL_corrected[i] = 0;
      for(int j = 0; j < dg_npf; j++) {
        if(fp_equal(x[0][fmaskL[i]], x[1][fmaskR[j]]) &&
           fp_equal(z[0][fmaskL[i]], z[1][fmaskR[j]]))
          fmaskR_corrected[i] = fmaskR[j];
        if(fp_equal(x[1][fmaskR[i]], x[0][fmaskL[j]]) &&
           fp_equal(z[1][fmaskR[i]], z[0][fmaskL[j]]))
          fmaskL_corrected[i] = fmaskL[j];
      }
    }

    for(int i = 0; i < dg_npf; i++) {
      if(!fp_equal(x[0][fmaskL[i]], x[1][fmaskR_corrected[i]]) ||
         !fp_equal(z[0][fmaskL[i]], z[1][fmaskR_corrected[i]]))
        (*num)++;
    }
  } else if(*periodicFace == 3) {
    for(int i = 0; i < dg_npf; i++) {
      fmaskR_corrected[i] = 0;
      fmaskL_corrected[i] = 0;
      for(int j = 0; j < dg_npf; j++) {
        if(fp_equal(x[0][fmaskL[i]], x[1][fmaskR[j]]) &&
           fp_equal(y[0][fmaskL[i]], y[1][fmaskR[j]]))
          fmaskR_corrected[i] = fmaskR[j];
        if(fp_equal(x[1][fmaskR[i]], x[0][fmaskL[j]]) &&
           fp_equal(y[1][fmaskR[i]], y[0][fmaskL[j]]))
          fmaskL_corrected[i] = fmaskL[j];
      }
    }

    for(int i = 0; i < dg_npf; i++) {
      if(!fp_equal(x[0][fmaskL[i]], x[1][fmaskR_corrected[i]]) ||
         !fp_equal(y[0][fmaskL[i]], y[1][fmaskR_corrected[i]]))
        (*num)++;
    }
  }
}
