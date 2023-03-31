inline void init_edges(const int *edgeNum, const DG_FP **x, const DG_FP **y,
                       bool *reverse) {
  int edgeL = edgeNum[0];
  int edgeR = edgeNum[1];

  // Check if periodic boundary edge
  if(fabs(x[0][0] - x[1][0]) > 2.0 || fabs(y[0][0] - y[1][0]) > 2.0) {
    // Check if in x or y direction
    if(fabs(x[0][0] - x[1][0]) > 2.0) {
      if(edgeR == 0) {
        if(edgeL == 0) {
          *reverse = !fp_equal(y[0][0], y[1][0]);
        } else if(edgeL == 1) {
          *reverse = !fp_equal(y[0][1], y[1][0]);
        } else {
          *reverse = !fp_equal(y[0][2], y[1][0]);
        }
      } else if(edgeR == 1) {
        if(edgeL == 0) {
          *reverse = !fp_equal(y[0][0], y[1][1]);
        } else if(edgeL == 1) {
          *reverse = !fp_equal(y[0][1], y[1][1]);
        } else {
          *reverse = !fp_equal(y[0][2], y[1][1]);
        }
      } else {
        if(edgeL == 0) {
          *reverse = !fp_equal(y[0][0], y[1][2]);
        } else if(edgeL == 1) {
          *reverse = !fp_equal(y[0][1], y[1][2]);
        } else {
          *reverse = !fp_equal(y[0][2], y[1][2]);
        }
      }
    } else {
      if(edgeR == 0) {
        if(edgeL == 0) {
          *reverse = !fp_equal(x[0][0], x[1][0]);
        } else if(edgeL == 1) {
          *reverse = !fp_equal(x[0][1], x[1][0]);
        } else {
          *reverse = !fp_equal(x[0][2], x[1][0]);
        }
      } else if(edgeR == 1) {
        if(edgeL == 0) {
          *reverse = !fp_equal(x[0][0], x[1][1]);
        } else if(edgeL == 1) {
          *reverse = !fp_equal(x[0][1], x[1][1]);
        } else {
          *reverse = !fp_equal(x[0][2], x[1][1]);
        }
      } else {
        if(edgeL == 0) {
          *reverse = !fp_equal(x[0][0], x[1][2]);
        } else if(edgeL == 1) {
          *reverse = !fp_equal(x[0][1], x[1][2]);
        } else {
          *reverse = !fp_equal(x[0][2], x[1][2]);
        }
      }
    }
  } else {
    // Regular edges
    if(edgeR == 0) {
      if(edgeL == 0) {
        *reverse = !(fp_equal(x[0][0], x[1][0]) && fp_equal(y[0][0], y[1][0]));
      } else if(edgeL == 1) {
        *reverse = !(fp_equal(x[0][1], x[1][0]) && fp_equal(y[0][1], y[1][0]));
      } else {
        *reverse = !(fp_equal(x[0][2], x[1][0]) && fp_equal(y[0][2], y[1][0]));
      }
    } else if(edgeR == 1) {
      if(edgeL == 0) {
        *reverse = !(fp_equal(x[0][0], x[1][1]) && fp_equal(y[0][0], y[1][1]));
      } else if(edgeL == 1) {
        *reverse = !(fp_equal(x[0][1], x[1][1]) && fp_equal(y[0][1], y[1][1]));
      } else {
        *reverse = !(fp_equal(x[0][2], x[1][1]) && fp_equal(y[0][2], y[1][1]));
      }
    } else {
      if(edgeL == 0) {
        *reverse = !(fp_equal(x[0][0], x[1][2]) && fp_equal(y[0][0], y[1][2]));
      } else if(edgeL == 1) {
        *reverse = !(fp_equal(x[0][1], x[1][2]) && fp_equal(y[0][1], y[1][2]));
      } else {
        *reverse = !(fp_equal(x[0][2], x[1][2]) && fp_equal(y[0][2], y[1][2]));
      }
    }
  }
}
