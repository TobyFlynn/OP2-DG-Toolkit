inline void copy_to_edges_ll(const DG_MAT_IND_TYPE **dat, DG_MAT_IND_TYPE *datL,
                             DG_MAT_IND_TYPE *datR) {
  datL[0] = dat[0][0];
  datR[0] = dat[1][0];
}
