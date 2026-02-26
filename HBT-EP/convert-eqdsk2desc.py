import sys

# this should be where you cloned the efit2desc repo
sys.path.insert(1, "../../../efit2desc")

# to make things work properly use a new environment
# for efit2desc
from efit2desc import (
    convert_EFIT_to_DESC,
    plot_eq_iota_against_efit,
    plot_eq_surfaces_against_efit,
)

name = "HBT_105995_06"
eqdsk_name = f"From-Jeff/{name}.eqdsk"

# I had to comment some saving functions in the source code
# if you give a relative path, internal file names becomes
# weird
eq, _ = convert_EFIT_to_DESC(
    eqdsk_name, L=16, M=16, psiN_cutoff=1.0, plot=False, save=False
)
eq.change_resolution(NFP=64)
eq.save(f"desc-eq-{name}.h5")
