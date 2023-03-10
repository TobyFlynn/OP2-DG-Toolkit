import os
import sys

gemv_template = \
"""
#ifndef OP2_DG_CUDA
#if DG_DOUBLE == 1
cblas_dgemv({row_col},{trans},{m},{n},{alpha},{A},{lda},{x}, 1,{beta},{y}, 1);
#else
cblas_sgemv({row_col},{trans},{m},{n},{alpha},{A},{lda},{x}, 1,{beta},{y}, 1);
#endif
#else
for(int i = 0; i < {outer}; i++) {{
  {y_init}
  for(int j = 0; j < {inner}; j++) {{
    int ind = DG_MAT_IND({ind_0},{ind_1},{m},{n});
    ({y})[i] += {alpha_mult} ({A})[ind] * ({x})[j];
  }}
}}
#endif
"""

def replace_gemv_kernels(input_str):
    out_str = input_str
    index = out_str.find("op2_in_kernel_gemv")
    col_maj = True
    while index != -1:
        end_ind = out_str.find(")", index)
        args_str = out_str[out_str.find("(", index) + 1 : end_ind]
        args_str = args_str.split(",")
        col_row_maj_str = "CblasColMajor"
        if not col_maj:
            col_row_maj_str = "CblasRowMajor"
        transpose_str = " CblasNoTrans"
        ind0 = "i"
        ind1 = "j"
        outer_l = args_str[1]
        inner_l = args_str[2]
        if "true" in args_str[0]:
            transpose_str = " CblasTrans"
            ind0 = "j"
            ind1 = "i"
            outer_l = args_str[2]
            inner_l = args_str[1]
        y_init_str = "({y})[i] *= {beta};".format(y = args_str[8], beta = args_str[7])
        if args_str[7].strip() == "0.0":
            y_init_str = "({y})[i] = 0.0;".format(y = args_str[8])
        alpha_mult_str = "({alpha}) * ".format(alpha = args_str[3])
        if args_str[3].strip() == "1.0":
            alpha_mult_str = ""
        blas_call = gemv_template.format(row_col = col_row_maj_str, trans = transpose_str, m = args_str[1], \
            n = args_str[2], alpha = args_str[3], A = args_str[4], lda = args_str[5], x = args_str[6], \
            beta = args_str[7], y = args_str[8], inner = inner_l, outer = outer_l, ind_0 = ind0, \
            ind_1 = ind1, y_init = y_init_str, alpha_mult = alpha_mult_str)
        out_str = out_str[0 : index] + blas_call + out_str[end_ind + 2 :]
        index = out_str.find("op2_in_kernel_gemv")
    return out_str


dim = sys.argv[1]
order = sys.argv[2]

dg_np        = "1"
dg_npf       = "1"
dg_cub_np    = "1"
dg_g_np      = "1"
dg_gf_np     = "1"
dg_order     = "1"
dg_num_faces = ""

fp_type = "d"

# Get DG order from command line args
if dim == "2":
    dg_order  = order
    order_int = int(order)
    dg_np     = str(int((order_int + 1) * (order_int + 2) / 2))
    dg_npf    = str(order_int + 1)
    dg_num_faces = "3"
    if order == "1":
        dg_cub_np    = "12"
        dg_g_np      = "9"
        dg_gf_np     = "3"
    elif order == "2":
        dg_cub_np    = "16"
        dg_g_np      = "12"
        dg_gf_np     = "4"
    elif order == "3":
        dg_cub_np    = "36"
        dg_g_np      = "18"
        dg_gf_np     = "6"
    elif order == "4":
        dg_cub_np    = "46"
        dg_g_np      = "21"
        dg_gf_np     = "7"
    else:
        print("This order of DG is not supported yet...exiting")
        sys.exit()
elif dim == "3":
    dg_order  = order
    order_int = int(order)
    dg_np     = str(int((order_int + 1) * (order_int + 2) * (order_int + 3) / 6))
    dg_npf    = str(int((order_int + 1) * (order_int + 2) / 2))
    dg_num_faces = "4"

inputfiles = []

for dirpath, _, filenames in os.walk("src"):
    for f in filenames:
        if f[0] != '.':
            tmp  = dirpath + "/" + f
            tmp2 = tmp.split("/")
            tmp3 = "/".join(tmp2[1:])
            inputfiles.append(tmp3)

for f in inputfiles:
    filedata = None
    with open("src/" + f, "r") as file:
        filedata = file.read()

    newdata = filedata
    newdata = replace_gemv_kernels(newdata)
    if "CMakeLists" not in f:
        if fp_type == "d":
            newdata = newdata.replace("DG_FP_STR", "\"double\"")
            newdata = newdata.replace("DG_FP", "double")
        else:
            newdata = newdata.replace("DG_FP_STR", "\"float\"")
            newdata = newdata.replace("DG_FP", "float")
        newdata = newdata.replace("DG_ORDER", dg_order)
        newdata = newdata.replace("DG_NPF", dg_npf)
        newdata = newdata.replace("DG_NP", dg_np)
        newdata = newdata.replace("DG_CUB_NP", dg_cub_np)
        newdata = newdata.replace("DG_G_NP", dg_g_np)
        newdata = newdata.replace("DG_GF_NP", dg_gf_np)
        newdata = newdata.replace("DG_NUM_FACES", dg_num_faces)

    if dim == "2":
        with open("gen_2d/" + f, "w") as file:
            file.write(newdata)
    elif dim == "3":
        with open("gen_3d/" + f, "w") as file:
            file.write(newdata)
