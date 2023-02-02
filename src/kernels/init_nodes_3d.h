inline void init_nodes_3d(const double **nc, double *nodeX, double *nodeY,
                          double *nodeZ) {
  nodeX[0] = nc[0][0];
  nodeX[1] = nc[1][0];
  nodeX[2] = nc[2][0];
  nodeX[3] = nc[3][0];
  nodeY[0] = nc[0][1];
  nodeY[1] = nc[1][1];
  nodeY[2] = nc[2][1];
  nodeY[3] = nc[3][1];
  nodeZ[0] = nc[0][2];
  nodeZ[1] = nc[1][2];
  nodeZ[2] = nc[2][2];
  nodeZ[3] = nc[3][2];
}