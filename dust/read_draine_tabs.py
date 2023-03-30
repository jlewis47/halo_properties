import numpy as np


def read_draine_tab(path):
    """
    lambda, albedo, g, C_ext/H(cm^2/H), K_abs(cm^2/g)
    """

    with open(path, "r") as src:

        line = ""
        for line in src:
            if "--" in line : break

        #table = np.genfromtxt(src)
        table = np.genfromtxt(" ".join(ln.split()[:5]) for ln in src)



    return(table)


def kappa_wvlngth(table, wvlngth):
    """
    wvlngth in ang
    """

    wvlngth *= 1e-4

    #print("Searching for %e microns"%wvlngth)

    assert wvlngth<=np.max(table[:,0]) and wvlngth>=np.min(table[:,0]), "Target wavelength not included in table"

    #find closest in table
    return(table[np.argmin(np.abs(wvlngth-table[:,0])),
    -1])