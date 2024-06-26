#ifndef __DG_COMPILER_DEFS_H
#define __DG_COMPILER_DEFS_H

#define DG_DOUBLE 1
#define DG_MAT_IND_LL 0

#if DG_DOUBLE == 1
#define DG_FP double
#define fp_equal(x,y) (fabs((x) - (y)) < 1e-8)
#else
#define DG_FP float
#define fp_equal(x,y) (fabs((x) - (y)) < 1e-6)
#endif

#if DG_MAT_IND_LL == 1
#define DG_MAT_IND_TYPE ll
#else
#define DG_MAT_IND_TYPE int
#endif

#ifdef DG_COL_MAJ
// i = row, j = col, m = total rows, n = total cols
#define DG_MAT_IND(i, j, m, n) ((i) + (j) * (m))
#else
#define DG_MAT_IND(i, j, m, n) ((i) * (n) + (j))
#endif

#if DG_DIM == 2
#define RX_IND 0
#define RY_IND 1
#define SX_IND 2
#define SY_IND 3
#define J_IND  4
#else
#define RX_IND 0
#define RY_IND 1
#define RZ_IND 2
#define SX_IND 3
#define SY_IND 4
#define SZ_IND 5
#define TX_IND 6
#define TY_IND 7
#define TZ_IND 8
#define J_IND  9
#endif

#if DG_DIM == 2

#define DG_NUM_FACES 3
#define DG_NUM_CONSTANTS 5

#if DG_ORDER == 7

// Number of points per triangle
#define DG_NP 36
// Number of points per face
#define DG_NPF 8

#elif DG_ORDER == 6

// Number of points per triangle
#define DG_NP 28
// Number of points per face
#define DG_NPF 7

#elif DG_ORDER == 5

// Number of points per triangle
#define DG_NP 21
// Number of points per face
#define DG_NPF 6

#elif DG_ORDER == 4

// Number of points per triangle
#define DG_NP 15
// Number of points per face
#define DG_NPF 5

#elif DG_ORDER == 3

// Number of points per triangle
#define DG_NP 10
// Number of points per face
#define DG_NPF 4

#elif DG_ORDER == 2

// Number of points per triangle
#define DG_NP 6
// Number of points per face
#define DG_NPF 3

#elif DG_ORDER == 1

// Number of points per triangle
#define DG_NP 3
// Number of points per face
#define DG_NPF 2

#endif

#elif DG_DIM == 3

#define DG_NUM_FACES 4
#define DG_NUM_CONSTANTS 2

#if DG_ORDER == 7

// Number of points per element
#define DG_NP 120
// Number of points per face
#define DG_NPF 36

#elif DG_ORDER == 6

// Number of points per element
#define DG_NP 84
// Number of points per face
#define DG_NPF 28

#elif DG_ORDER == 5

// Number of points per element
#define DG_NP 56
// Number of points per face
#define DG_NPF 21

#elif DG_ORDER == 4

// Number of points per element
#define DG_NP 35
// Number of points per face
#define DG_NPF 15

#elif DG_ORDER == 3

// Number of points per element
#define DG_NP 20
// Number of points per face
#define DG_NPF 10

#elif DG_ORDER == 2

// Number of points per element
#define DG_NP 10
// Number of points per face
#define DG_NPF 6

#elif DG_ORDER == 1

// Number of points per element
#define DG_NP 4
// Number of points per face
#define DG_NPF 3

#endif

#endif

#endif
