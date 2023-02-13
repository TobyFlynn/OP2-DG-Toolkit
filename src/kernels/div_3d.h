inline void div_3d(const DG_FP *dr, const DG_FP *ds, const DG_FP *dt,
                const DG_FP *rc, const DG_FP *sc, const DG_FP *tc,
                DG_FP *res) {
  for(int i = 0; i < DG_NP; i++) {
    res[i] += *rc * dr[i] + *sc * ds[i] + *tc * dt[i];
  }
}