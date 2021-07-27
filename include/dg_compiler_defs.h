#ifndef __DG_COMPILER_DEFS_H
#define __DG_COMPILER_DEFS_H

#if DG_ORDER == 4
// Number of points per triangle
#define DG_NP 15
// Number of cubature points per triangle
#define DG_CUB_NP 46
// Number of gauss points per triangle
#define DG_G_NP 21
// Number of gauss points per face
#define DG_GF_NP 7
#elif DG_ORDER == 3
// Number of points per triangle
#define DG_NP 10
// Number of cubature points per triangle
#define DG_CUB_NP 36
// Number of gauss points per triangle
#define DG_G_NP 18
// Number of gauss points per face
#define DG_GF_NP 6
#endif

#endif
