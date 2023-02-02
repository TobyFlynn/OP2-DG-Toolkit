import os
import sys

dg_np        = ""
dg_npf       = ""

# Get DG order from command line args
if sys.argv[1] == "1":
    dg_order     = "1"
    dg_np        = "4"
    dg_npf       = "3"
elif sys.argv[1] == "2":
    dg_order     = "2"
    dg_np        = "10"
    dg_npf       = "6"
elif sys.argv[1] == "3":
    dg_order     = "3"
    dg_np        = "20"
    dg_npf       = "10"
elif sys.argv[1] == "4":
    dg_order     = "4"
    dg_np        = "35"
    dg_npf       = "15"
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

    with open("gen/" + f, "w") as file:
        file.write(newdata)