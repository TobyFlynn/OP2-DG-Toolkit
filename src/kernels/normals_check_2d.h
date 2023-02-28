inline void normals_check_2d(const int *edgeNum, const bool *rev, const DG_FP **nx,
                             const DG_FP **ny, const DG_FP **x, const DG_FP **y,
                             const DG_FP **nodeX_, const DG_FP **nodeY_, int *errors) {
  // Work out which edge for each element
  int edgeL = edgeNum[0];
  int edgeR = edgeNum[1];
  bool reverse = *rev;

  // Copy data from R to L
  int exIndL = edgeL * DG_GF_NP;
  int exIndR = edgeR * DG_GF_NP;

  DG_FP av_x = 0.0;
  DG_FP av_y = 0.0;
  for(int i = 0; i < DG_GF_NP; i++) {
    av_x += x[0][exIndL + i];
    av_y += y[0][exIndL + i];
  }
  av_x /= (DG_FP)DG_GF_NP;
  av_y /= (DG_FP)DG_GF_NP;

  DG_FP ptX = av_x + 1e-5 * nx[0][exIndL];
  DG_FP ptY = av_y + 1e-5 * ny[0][exIndL];

  DG_FP d1 = (ptX - nodeX_[0][1]) * (nodeY_[0][0] - nodeY_[0][1]) - (nodeX_[0][0] - nodeX_[0][1]) * (ptY - nodeY_[0][1]);
  DG_FP d2 = (ptX - nodeX_[0][2]) * (nodeY_[0][1] - nodeY_[0][2]) - (nodeX_[0][1] - nodeX_[0][2]) * (ptY - nodeY_[0][2]);
  DG_FP d3 = (ptX - nodeX_[0][0]) * (nodeY_[0][2] - nodeY_[0][0]) - (nodeX_[0][2] - nodeX_[0][0]) * (ptY - nodeY_[0][0]);

  bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
  bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

  if(!(has_neg && has_pos)) {
    *errors += 1;
  }

  av_x = 0.0;
  av_y = 0.0;
  for(int i = 0; i < DG_GF_NP; i++) {
    av_x += x[1][exIndR + i];
    av_y += y[1][exIndR + i];
  }
  av_x /= (DG_FP)DG_GF_NP;
  av_y /= (DG_FP)DG_GF_NP;

  ptX = av_x + 1e-5 * nx[1][exIndR];
  ptY = av_y + 1e-5 * ny[1][exIndR];

  d1 = (ptX - nodeX_[1][1]) * (nodeY_[1][0] - nodeY_[1][1]) - (nodeX_[1][0] - nodeX_[1][1]) * (ptY - nodeY_[1][1]);
  d2 = (ptX - nodeX_[1][2]) * (nodeY_[1][1] - nodeY_[1][2]) - (nodeX_[1][1] - nodeX_[1][2]) * (ptY - nodeY_[1][2]);
  d3 = (ptX - nodeX_[1][0]) * (nodeY_[1][2] - nodeY_[1][0]) - (nodeX_[1][2] - nodeX_[1][0]) * (ptY - nodeY_[1][0]);

  has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
  has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

  if(!(has_neg && has_pos)) {
    *errors += 1;
  }

  if(!fp_equal(nx[0][exIndL], -nx[1][exIndR]) || !fp_equal(ny[0][exIndL], -ny[1][exIndR]))
    *errors += 1;
}
