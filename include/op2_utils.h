#ifndef __OP2_UTILS_H
#define __OP2_UTILS_H

#include "op_seq.h"

DG_FP *getOP2PtrDevice(op_dat dat, op_access acc);
void releaseOP2PtrDevice(op_dat dat, op_access acc, const DG_FP *ptr);
DG_FP *getOP2PtrHost(op_dat dat, op_access acc);
void releaseOP2PtrHost(op_dat dat, op_access acc, const DG_FP *ptr);

DG_FP *getOP2PtrDeviceHE(op_dat dat, op_access acc);
void releaseOP2PtrDeviceHE(op_dat dat, op_access acc, const DG_FP *ptr);
DG_FP *getOP2PtrHostHE(op_dat dat, op_access acc);
void releaseOP2PtrHostHE(op_dat dat, op_access acc, const DG_FP *ptr);

#endif
