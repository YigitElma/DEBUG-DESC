import numpy as np


def print_modes_coefs(thing, coord):
    """Print mode numbers and coefficients as a table.

    Parameters
    ----------
    thing : equilibrium or surface
        object to print
    coord : string
        Coordinate to print. It can be only "R", "Z" or "L"

    """
    if coord == "R":
        basis = thing.R_basis
        coeff = thing.R_lmn
        name = "R_lmn = "
    elif coord == "Z":
        basis = thing.Z_basis
        coeff = thing.Z_lmn
        name = "Z_lmn = "
    elif coord == "L":
        basis = thing.L_basis
        coeff = thing.L_lmn
        name = "L_lmn = "
    else:
        raise ValueError(
            f"Coordinate can only be 'R', 'Z' or 'L'. Given value is {coord}"
        )

    table = np.empty([basis.num_modes, basis.num_modes + 1])
    table[:, :3] = basis.modes
    table[:, 3] = coeff
    print(
        f"Printing {basis.num_modes} results for {coord} coordinate of "
        + f"{thing.__class__.__name__}"
    )
    print("-------------------------------------")
    for row in table:
        print(
            "{:}{:<{w}.0f} {:}{:<{w}.0f} {:}{:<{w}.0f} {:}{:>{w2}.8f}".format(
                "l = ", row[0], "m = ", row[1], "n = ", row[2], name, row[3], w=4, w2=12
            )
        )
