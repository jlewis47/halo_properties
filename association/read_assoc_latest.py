import numpy as np
import string
import argparse
import os
from ..utils.output_paths import *
from ..utils.utils import *

# from ..files.read_stars import unpair, read_star_file, read_specific_stars
import h5py


def read_assoc(
    out_nb,
    sim_name,
    ldx,
    sub_ldx=None,
    rtwo_fact=1,
    ll=200,
    assoc_mthd="",
    subnb=None,
    bounds=None,
    mass_cut=None,
    mp=False,
):
    """
    Get right binary files, format correctly and return
    rel_fof_path is the relative path to the fof files of each snapshot starting from the snapshot directory

    """

    """
    Open association files
    """

    check_assoc_keys(assoc_mthd)

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix=fof_suffix, rtwo_suffix=rtwo_suffix, mp=mp)

    print("suffix:", suffix)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    # print(rps)
    if subnb != None:
        rps = int(ldx / sub_ldx)
        ix, iy, iz = np.unravel_index(int(subnb), (rps, rps, rps))

    fname = os.path.join(assoc_out, ("assoc_halos_%s" % out_nb) + suffix)
    print(fof_suffix, rtwo_suffix, suffix)
    print("looking for:", fname)

    with h5py.File(fname, "r") as F:
        # keys = list(F.keys())
        # ids = F["ID"][()]

        has_stars = F["stellar count"][()] > 0

        coords = np.where(
            np.tile(has_stars, (3, 1)).T,
            F["coords_new"][()][:, :],
            F["coords"][()][:, :] * ldx,
        )

        x, y, z = coords.T

        if subnb != None or bounds != None:
            if subnb != None:
                cond = (
                    ((x >= ix * sub_ldx) * (x < (ix + 1.0) * sub_ldx))
                    * ((y >= iy * sub_ldx) * (y < (iy + 1.0) * sub_ldx))
                    * ((z >= iz * sub_ldx) * (z < (iz + 1.0) * sub_ldx))
                )
            else:
                cond = (
                    (x >= bounds[0][0])
                    * (x < bounds[0][1])
                    * (y >= bounds[1][0])
                    * (y < bounds[1][1])
                    * (z >= bounds[2][0])
                    * (z < bounds[2][1])
                )

        else:
            cond = np.full(x.shape, True)

        # print(np.min(F["mass"]),np.max(F["mass"]),mass_cut)

        if mass_cut != None:
            mass_cond = F["mass"][()] > mass_cut

            cond = cond * mass_cond

        l = np.count_nonzero(cond)
        fofs = np.empty(
            (l),
            dtype=[
                ("ids", "i8"),
                # ("fnb", "i4"),
                ("mass", "f4"),
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("rpx", "f4"),
                ("mstar", "f4"),
                ("nstar", "i4"),
            ],
        )

        # apply mass and pos cut
        fofs["mass"] = F["mass"][cond]
        # fofs["fnb"] = F["fnb"][cond]
        fofs["x"] = x[cond]
        fofs["y"] = y[cond]
        fofs["z"] = z[cond]
        fofs["ids"] = F["ID"][cond]
        fofs["rpx"] = F["rpx"][cond]
        fofs["mstar"] = F["stellar mass"][cond]
        fofs["nstar"] = F["stellar count"][cond]

        halo_star_ids = F["halo star ID"][()]

        cum_star_nb = np.cumsum(F["stellar count"])[cond]

    return (fofs, halo_star_ids, cum_star_nb)
