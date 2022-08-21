import os
import sys

dg_np        = ""
dg_npf       = ""
dg_cub_np    = ""
dg_g_np      = ""
dg_gf_np     = ""

# Get DG order from command line args
if sys.argv[1] == "1":
    dg_order     = "1"
    dg_np        = "3"
    dg_npf       = "2"
    dg_cub_np    = "12"
    dg_g_np      = "9"
    dg_gf_np     = "3"
elif sys.argv[1] == "2":
    dg_order     = "2"
    dg_np        = "6"
    dg_npf       = "3"
    dg_cub_np    = "16"
    dg_g_np      = "12"
    dg_gf_np     = "4"
elif sys.argv[1] == "3":
    dg_order     = "3"
    dg_np        = "10"
    dg_npf       = "4"
    dg_cub_np    = "36"
    dg_g_np      = "18"
    dg_gf_np     = "6"
elif sys.argv[1] == "4":
    dg_order     = "4"
    dg_np        = "15"
    dg_npf       = "5"
    dg_cub_np    = "46"
    dg_g_np      = "21"
    dg_gf_np     = "7"
else:
    print("This order of DG is not supported yet...exiting")
    sys.exit()

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
    if "CMakeLists" not in f:
        newdata = newdata.replace("DG_ORDER", dg_order)
        newdata = newdata.replace("DG_NPF", dg_npf)
        newdata = newdata.replace("DG_NP", dg_np)
        newdata = newdata.replace("DG_CUB_NP", dg_cub_np)
        newdata = newdata.replace("DG_G_NP", dg_g_np)
        newdata = newdata.replace("DG_GF_NP", dg_gf_np)

    with open("gen/" + f, "w") as file:
        file.write(newdata)
