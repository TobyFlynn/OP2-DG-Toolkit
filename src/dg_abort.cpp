#include "dg_abort.h"

#ifdef DG_MPI
#include "mpi.h"
#include <cstdio>
#else
#include <stdexcept>
#endif

void dg_abort(const std::string &msg) {
#ifdef DG_MPI
  fprintf(stderr, "%s\n", msg.c_str());
  MPI_Abort(MPI_COMM_WORLD, 0);
#else
  throw std::runtime_error(msg);
#endif
}