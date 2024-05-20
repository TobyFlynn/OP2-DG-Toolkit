import os
import sys

def get_2d_constants(order):
    order_int = int(order)
    dg_np     = str(int((order_int + 1) * (order_int + 2) / 2))
    dg_npf    = str(order_int + 1)
    if order == "1":
        dg_cub_np = "3"
        dg_cub_surf_np = "3"
    elif order == "2":
        dg_cub_np = "6"
        dg_cub_surf_np = "5"
    elif order == "3":
        dg_cub_np = "12"
        dg_cub_surf_np = "7"
    elif order == "4":
        dg_cub_np = "16"
        dg_cub_surf_np = "9"
    elif order == "5":
        dg_cub_np = "25"
        dg_cub_surf_np = "11"
    elif order == "6":
        dg_cub_np = "36"
        dg_cub_surf_np = "13"
    elif order == "7":
        dg_cub_np = "46"
        dg_cub_surf_np = "15"
    else:
        print("This order of DG is not supported yet...exiting")
        sys.exit()
    return dg_np, dg_npf, dg_cub_np, dg_cub_surf_np

def get_3d_constants(order):
    order_int = int(order)
    dg_np     = str(int((order_int + 1) * (order_int + 2) * (order_int + 3) / 6))
    dg_npf    = str(int((order_int + 1) * (order_int + 2) / 2))
    if order == "1":
        dg_cub_np = "4"
        dg_cub_surf_np = "3"
    elif order == "2":
        dg_cub_np = "11"
        dg_cub_surf_np = "6"
    elif order == "3":
        dg_cub_np = "23"
        dg_cub_surf_np = "12"
    elif order == "4":
        dg_cub_np = "44"
        dg_cub_surf_np = "16"
    elif order == "5":
        dg_cub_np = "74"
        dg_cub_surf_np = "25"
    elif order == "6":
        dg_cub_np = "122"
        dg_cub_surf_np = "36"
    elif order == "7":
        dg_cub_np = "177"
        dg_cub_surf_np = "46"
    else:
        print("This order of DG is not supported yet...exiting")
        sys.exit()
    return dg_np, dg_npf, dg_cub_np, dg_cub_surf_np

gemv_template = \
"""
#if !defined(OP2_DG_CUDA) && !defined(OP2_DG_HIP)
#if DG_DOUBLE == 1
cblas_dgemv({row_col},{trans},{m},{n},{alpha},{A},{lda},{x}, 1,{beta},{y}, 1);
#else
cblas_sgemv({row_col},{trans},{m},{n},{alpha},{A},{lda},{x}, 1,{beta},{y}, 1);
#endif
#else
for(int i = 0; i < {outer}; i++) {{
  DG_FP tmp_gemv = 0.0;
  for(int j = 0; j < {inner}; j++) {{
    int ind = DG_MAT_IND({ind_0},{ind_1},{m},{n});
    tmp_gemv += {A}[ind] * {x}[j];
  }}
  {y_update}
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
        y_update_str = "{y}[i] = ({beta}) * {y}[i] + ({alpha}) * tmp_gemv;".format(y = args_str[8], alpha = args_str[3], beta = args_str[7])
        if args_str[7].strip() == "0.0":
            y_update_str = "{y}[i] = ({alpha}) * tmp_gemv;".format(y = args_str[8], alpha = args_str[3])
        blas_call = gemv_template.format(row_col = col_row_maj_str, trans = transpose_str, m = args_str[1], \
            n = args_str[2], alpha = args_str[3], A = args_str[4], lda = args_str[5], x = args_str[6], \
            beta = args_str[7], y = args_str[8], inner = inner_l, outer = outer_l, ind_0 = ind0, \
            ind_1 = ind1, y_update = y_update_str)
        out_str = out_str[0 : index] + blas_call + out_str[end_ind + 2 :]
        index = out_str.find("op2_in_kernel_gemv")
    return out_str

gemm_template = \
"""
#if !defined(OP2_DG_CUDA) && !defined(OP2_DG_HIP)
#if DG_DOUBLE == 1
cblas_dgemm({row_col}, {transA}, {transB}, {m}, {n}, {k}, {alpha}, {A}, {lda}, {B}, {ldb}, {beta}, {C}, {ldc});
#else
cblas_sgemm({row_col}, {transA}, {transB}, {m}, {n}, {k}, {alpha}, {A}, {lda}, {B}, {ldb}, {beta}, {C}, {ldc});
#endif
#else
for(int i = 0; i < {m}; i++) {{
  for(int j = 0; j < {n}; j++) {{
    DG_FP tmp_gemm = 0.0;
    for(int k = 0; k < {k}; k++) {{
      int a_ind = DG_MAT_IND({indA_0}, {indA_1}, {m}, {k});
      int b_ind = DG_MAT_IND({indB_0}, {indB_1}, {k}, {n});
      tmp_gemm += {A}[a_ind] * {B}[b_ind];
    }}
    int c_ind = DG_MAT_IND(i, j, {m}, {n});
    {c_update}
  }}
}}
#endif
"""

def replace_gemm_kernels(input_str):
    out_str = input_str
    index = out_str.find("op2_in_kernel_gemm")
    col_maj = True
    while index != -1:
        end_ind = out_str.find(")", index)
        args_str = out_str[out_str.find("(", index) + 1 : end_ind]
        args_str = args_str.split(",")
        col_row_maj_str = "CblasColMajor"
        if not col_maj:
            col_row_maj_str = "CblasRowMajor"
        transpose_strA = " CblasNoTrans"
        indA0 = "i"
        indA1 = "k"
        if "true" in args_str[0]:
            transpose_strA = " CblasTrans"
            indA0 = "k"
            indA1 = "i"
        transpose_strB = " CblasNoTrans"
        indB0 = "k"
        indB1 = "j"
        if "true" in args_str[1]:
            transpose_strB = " CblasTrans"
            indB0 = "j"
            indB1 = "k"
        c_update_str = "{C}[c_ind] = ({beta}) * {C}[c_ind] + ({alpha}) * tmp_gemm;".format(C = args_str[11], beta = args_str[10], alpha = args_str[5])
        if args_str[10].strip() == "0.0":
            c_update_str = "{C}[c_ind] = ({alpha}) * tmp_gemm;".format(C = args_str[11], alpha = args_str[5])
        blas_call = gemm_template.format(row_col = col_row_maj_str, transA = transpose_strA, transB = transpose_strB, \
                        m = args_str[2], n = args_str[3], k = args_str[4], alpha = args_str[5], A = args_str[6], \
                        lda = args_str[7], B = args_str[8], ldb = args_str[9], beta = args_str[10], C = args_str[11], \
                        ldc = args_str[12], indA_0 = indA0, indA_1 = indA1, indB_0 = indB0, indB_1 = indB1, \
                        c_update = c_update_str)
        out_str = out_str[0 : index] + blas_call + out_str[end_ind + 2 :]
        index = out_str.find("op2_in_kernel_gemm")
    return out_str


dim = sys.argv[1]
order = sys.argv[2]

dg_np        = "1"
dg_npf       = "1"
dg_order     = "1"
dg_num_faces = "1"
dg_np_n1     = "1"
dg_npf_n1    = "1"
dg_cub_3d_np = "1"
dg_cub_surf_3d_np = "1"
dg_cub_2d_np = "1"
dg_cub_surf_2d_np = "1"

fp_type = "d"
mat_int_type = 'int'
#mat_int_type = 'll'

# Get DG order from command line args
if dim == "2":
    dg_order  = order
    dg_np, dg_npf, dg_cub_2d_np, dg_cub_surf_2d_np = get_2d_constants(order)
    dg_np_n1, dg_npf_n1 = get_2d_constants("1")[0:2]
    dg_num_faces = "3"
elif dim == "3":
    dg_order  = order
    order_int = int(order)
    dg_np, dg_npf, dg_cub_3d_np, dg_cub_surf_3d_np = get_3d_constants(order)
    dg_np_n1, dg_npf_n1 = get_3d_constants("1")[0:2]
    dg_num_faces = "4"

#######################################
# Generate replacement for BLAS stubs #
#######################################

# Get matrix sizes that will fit in shared memory
def get_valid_matrix_sizes(max_order, max_shared_mem_bytes, num_bytes_per_element):
    matrix_sizes = []
    # Get matrix sizes within each DG order
    for order in range(1, max_order + 1):
        np = ""
        npf = ""
        if dim == "2":
            np, npf = get_2d_constants(str(order))[:2]
        elif dim == "3":
            np, npf = get_3d_constants(str(order))[:2]
        np = int(np)
        npf = int(npf)
        if np * np * num_bytes_per_element <= max_shared_mem_bytes:
            matrix_sizes.append((np, np))
        if np * npf * int(dg_num_faces) * num_bytes_per_element <= max_shared_mem_bytes:
            matrix_sizes.append((np, npf * int(dg_num_faces)))

    # Get matrix sizes for over integration matrices
    np = ""
    npf = ""
    cub_np = ""
    cub_surf_np = ""
    if dim == "2":
        np, npf, cub_np, cub_surf_np = get_2d_constants(str(max_order))
    elif dim == "3":
        np, npf, cub_np, cub_surf_np = get_3d_constants(str(max_order))
    np = int(np)
    npf = int(npf)
    cub_np = int(cub_np)
    cub_surf_np = int(cub_surf_np)
    if np * cub_np * num_bytes_per_element <= max_shared_mem_bytes:
        matrix_sizes.append((cub_np, np))
        matrix_sizes.append((np, cub_np))
    if np * cub_surf_np * int(dg_num_faces) * num_bytes_per_element <= max_shared_mem_bytes:
        matrix_sizes.append((cub_surf_np * int(dg_num_faces), np))
        matrix_sizes.append((np, cub_surf_np * int(dg_num_faces)))

    # Get matrix sizes for interpolation between DG orders
    for order_0 in range(1, max_order + 1):
        for order_1 in range(order_0 + 1, max_order + 1):
            np_0 = ""
            np_1 = ""
            if dim == "2":
                np_0 = get_2d_constants(str(order_0))[0]
                np_1 = get_2d_constants(str(order_1))[0]
            elif dim == "3":
                np_0 = get_3d_constants(str(order_0))[0]
                np_1 = get_3d_constants(str(order_1))[0]
            np_0 = int(np_0)
            np_1 = int(np_1)
            if np_0 * np_1 * num_bytes_per_element <= max_shared_mem_bytes:
                matrix_sizes.append((np_0, np_1))
                matrix_sizes.append((np_1, np_0))
    return matrix_sizes

# CUDA custom kernel calls
gpu_custom_blas_template = \
"""if(m == {row} && n == {col}) {{
  templated_wrapper_{type_postfix}<{row},{col}>(t, nblocks, nthread, n*m*sizeof({type}),
                            strideX, strideY, alpha, beta, matrix,
                            x_ptr, y_ptr, num_vecs);
}}"""
def generate_custom_gpu_soa_blas_calls(max_order, max_shared_mem_bytes, double_precision):
    num_bytes_per_element = 8 # doubles
    if not double_precision:
        num_bytes_per_element = 4 # floats
    matrix_sizes = get_valid_matrix_sizes(max_order, max_shared_mem_bytes, num_bytes_per_element)

    # Generate templated function calls
    type = "double"
    type_postfix = "dp"
    if not double_precision:
        type = "float"
        type_postfix = "sp"
    code = ""
    first = True
    for matrix in matrix_sizes:
        if not first:
            code = code + " else "
        code = code + gpu_custom_blas_template.format(row = matrix[0], col = matrix[1], type_postfix = type_postfix, type = type)
        first = False
    return code

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
    newdata = replace_gemm_kernels(newdata)
    if "CMakeLists" not in f:
        if mat_int_type == "int":
            newdata = newdata.replace("DG_MAT_IND_TYPE_STR", "\"int\"")
            newdata = newdata.replace("DG_MAT_IND_TYPE", "int")
        else:
            newdata = newdata.replace("DG_MAT_IND_TYPE_STR", "\"ll\"")
            newdata = newdata.replace("DG_MAT_IND_TYPE", "ll")
        if fp_type == "d":
            newdata = newdata.replace("DG_FP_STR", "\"double\"")
            newdata = newdata.replace("DG_FP", "double")
            newdata = newdata.replace("DG_MPI_FP", "MPI_DOUBLE")
        else:
            newdata = newdata.replace("DG_FP_STR", "\"float\"")
            newdata = newdata.replace("DG_FP", "float")
            newdata = newdata.replace("DG_MPI_FP", "MPI_FLOAT")
        newdata = newdata.replace("DG_ORDER", dg_order)
        newdata = newdata.replace("DG_NPF_N1", dg_npf_n1)
        newdata = newdata.replace("DG_NPF", dg_npf)
        newdata = newdata.replace("DG_NP_N1", dg_np_n1)
        newdata = newdata.replace("DG_NP", dg_np)
        newdata = newdata.replace("DG_NUM_FACES", dg_num_faces)
        newdata = newdata.replace("DG_CUB_3D_NP", dg_cub_3d_np)
        newdata = newdata.replace("DG_CUB_SURF_3D_NP", dg_cub_surf_3d_np)
        newdata = newdata.replace("DG_CUB_2D_NP", dg_cub_2d_np)
        newdata = newdata.replace("DG_CUB_SURF_2D_NP", dg_cub_surf_2d_np)

        max_shared_mem_bytes = 64 * 1024 # TODO confirm
        gpu_dp_custom_kernel_calls = generate_custom_gpu_soa_blas_calls(int(dg_order), max_shared_mem_bytes, True)
        gpu_sp_custom_kernel_calls = generate_custom_gpu_soa_blas_calls(int(dg_order), max_shared_mem_bytes, False)
        newdata = newdata.replace("[OP2_DG_GPU_SOA_DP_BLAS_STUB]", gpu_dp_custom_kernel_calls)
        newdata = newdata.replace("[OP2_DG_GPU_SOA_SP_BLAS_STUB]", gpu_sp_custom_kernel_calls)

    if dim == "2":
        with open("code_gen/gen_2d/" + f, "w") as file:
            file.write(newdata)
    elif dim == "3":
        with open("code_gen/gen_3d/" + f, "w") as file:
            file.write(newdata)
