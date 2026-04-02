import sys
import os

# this should be where you cloned the efit2desc repo
sys.path.insert(1, "../../../efit2desc")
sys.path.append(os.path.abspath("../../"))

# from desc import set_device

# set_device("gpu")

# to make things work properly use a new environment
# for efit2desc
from efit2desc import (
    convert_EFIT_to_DESC,
    plot_eq_iota_against_efit,
    plot_eq_surfaces_against_efit,
)
import numpy as np

name = "hbt_free_boundary2"
eqdsk_name = f"tokamaker/{name}.eqdsk"

# I had to comment some saving functions in the source code
# if you give a relative path, internal file names becomes
# weird
# eq, _ = convert_EFIT_to_DESC(
#     eqdsk_name,
#     L=24,
#     M=24,
#     psiN_cutoff=1.0,
#     plot=False,
#     save=False,
#     solve=False,
# )
# eq.change_resolution(NFP=64)
# eq.save(f"desc-eq-{name}-3.h5")

from omfit_classes import omfit_eqdsk

efit = omfit_eqdsk.OMFITgeqdsk(eqdsk_name)
# populate aux and flux-surface quantities
efit.addAuxQuantities()
efit.addFluxSurfaces(levels=list(np.linspace(0, 1, 129)))
# surfAvg computes safety factor, pressure, etc. on each flux surface
efit["fluxSurfaces"].surfAvg()


def list_keys(data, lvl=1):
    for key, value in data.items():
        if isinstance(value, dict):
            list_keys(value, lvl=lvl + 1)
        else:
            arrow = "--" * lvl
            print(f"{arrow}>{key}")


list_keys(efit, lvl=1)
