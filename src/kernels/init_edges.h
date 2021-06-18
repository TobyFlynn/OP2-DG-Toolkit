inline void init_edges(const int *edgeNum, const double **x, const double **y,
                       bool *reverse) {
  int edgeL = edgeNum[0];
  int edgeR = edgeNum[1];

  if(edgeR == 0) {
    if(edgeL == 0) {
      *reverse = !(x[0][0] == x[1][0] && y[0][0] == y[1][0]);
    } else if(edgeL == 1) {
      *reverse = !(x[0][1] == x[1][0] && y[0][1] == y[1][0]);
    } else {
      *reverse = !(x[0][2] == x[1][0] && y[0][2] == y[1][0]);
    }
  } else if(edgeR == 1) {
    if(edgeL == 0) {
      *reverse = !(x[0][0] == x[1][1] && y[0][0] == y[1][1]);
    } else if(edgeL == 1) {
      *reverse = !(x[0][1] == x[1][1] && y[0][1] == y[1][1]);
    } else {
      *reverse = !(x[0][2] == x[1][1] && y[0][2] == y[1][1]);
    }
  } else {
    if(edgeL == 0) {
      *reverse = !(x[0][0] == x[1][2] && y[0][0] == y[1][2]);
    } else if(edgeL == 1) {
      *reverse = !(x[0][1] == x[1][2] && y[0][1] == y[1][2]);
    } else {
      *reverse = !(x[0][2] == x[1][2] && y[0][2] == y[1][2]);
    }
  }
}
