#ifndef __DG_COMPILER_DEFS_H
#define __DG_COMPILER_DEFS_H

#if DG_DIM == 2

#define DG_NUM_FACES 3
#define DG_NUM_CONSTANTS 5

#if DG_ORDER == 4

// Number of points per triangle
#define DG_NP 15
// Number of points per face
#define DG_NPF 5
// Number of cubature points per triangle
#define DG_CUB_NP 46
// Number of gauss points per triangle
#define DG_G_NP 21
// Number of gauss points per face
#define DG_GF_NP 7

#elif DG_ORDER == 3

// Number of points per triangle
#define DG_NP 10
// Number of points per face
#define DG_NPF 4
// Number of cubature points per triangle
#define DG_CUB_NP 36
// Number of gauss points per triangle
#define DG_G_NP 18
// Number of gauss points per face
#define DG_GF_NP 6

#elif DG_ORDER == 2

// Number of points per triangle
#define DG_NP 6
// Number of points per face
#define DG_NPF 3
// Number of cubature points per triangle
#define DG_CUB_NP 16
// Number of gauss points per triangle
#define DG_G_NP 12
// Number of gauss points per face
#define DG_GF_NP 4

#elif DG_ORDER == 1

// Number of points per triangle
#define DG_NP 3
// Number of points per face
#define DG_NPF 2
// Number of cubature points per triangle
#define DG_CUB_NP 12
// Number of gauss points per triangle
#define DG_G_NP 9
// Number of gauss points per face
#define DG_GF_NP 3

#endif

#elif DG_DIM == 3

#define DG_NUM_FACES 4
#define DG_NUM_CONSTANTS 2

#if DG_ORDER == 4

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
