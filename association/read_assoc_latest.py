import numpy as np
import string
import argparse
import os
from ..utils.output_paths import *
from ..utils.utils import *
from ..params.params import ldx

# from ..files.read_stars import unpair, read_star_file, read_specific_stars
import h5py


def read_assoc(
    out_nb,
    sim_name,
    dset,
    sub_ldx=None,
    subnb=None,
    bounds=None,
    mass_cut=None,
    st_mass_cut=None,
    return_keys=None,
):
    """
    Get right binary files, format correctly and return
    rel_fof_path is the relative path to the fof files of each snapshot starting from the snapshot directory

    bounds is formatted as [[xmin,xmax],[ymin,ymax],[zmin,zmax]]


    """

    """
    Open association files
    """

    # mp = dset.mp
    clean = dset.clean
    # ll = dset.ll
    # rtwo_fact = dset.r200
    assoc_mthd = dset.assoc_mthd

    check_assoc_keys(assoc_mthd)

    out, assoc_out, analy_out, suffix = gen_paths(sim_name, out_nb, dset)

    # print(rps)
    if subnb != None:
        rps = int(ldx / sub_ldx)
        ix, iy, iz = np.unravel_index(int(subnb), (rps, rps, rps))

    if clean:
        fname = os.path.join(assoc_out, (f"assoc_halos_clean_{out_nb:d}") + suffix)
    else:
        fname = os.path.join(assoc_out, (f"assoc_halos_{out_nb:d}") + suffix)

    # print(fof_suffix, rtwo_suffix, suffix)
    # print("looking for:", fname)

    with h5py.File(fname, "r") as F:
        # keys = list(F.keys())
        # ids = F["ID"][()]

        has_stars = F["stellar count"][()] > 0

        if "coords_new" in F.keys() and "coords" in F.keys():
            coords = np.where(
                np.tile(has_stars, (3, 1)).T,
                F["coords_new"][()][:, :],
                F["coords"][()][:, :] * ldx,
            )
        else:
            coords = F["coords"][()]

        # print(fname, F.keys())

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

            single_val = np.prod(mass_cut) == mass_cut

            if single_val:

                mass_cond = F["mass"][()] > mass_cut

            else:
                min_mass, max_mass = mass_cut
                mass_cond = (F["mass"][()] > min_mass) * (F["mass"][()] < max_mass)

            cond = cond * mass_cond

        if st_mass_cut != None:

            single_val = np.all(np.prod(st_mass_cut) == st_mass_cut)

            if single_val:

                mass_cond = F["stellar mass"][()] > st_mass_cut

            else:
                min_mass, max_mass = st_mass_cut

                mass_cond = (F["stellar mass"][()] > min_mass) * (
                    F["stellar mass"][()] < max_mass
                )

            cond = cond * mass_cond

        return_types = {
            "ids": "i8",
            "mass": "f4",
            "x": "f8",
            "y": "f8",
            "z": "f8",
            "rpx": "f4",
            "mstar": "f4",
            "nstar": "i4",
        }
        read_2_return = {
            "ids": "ID",
            "mass": "mass",
            "rpx": "rpx",
            "mstar": "stellar mass",
            "nstar": "stellar count",
        }
        if return_keys == None:
            return_keys = ["ids", "mass", "x", "y", "z", "rpx", "mstar", "nstar"]

        return_dtype = []
        for key in return_keys:
            return_dtype.append((key, return_types[key]))

        l = np.count_nonzero(cond)
        fofs = np.empty(
            (l),
            return_dtype,
        )

        for key in return_keys:
            if key in read_2_return.keys():
                fofs[key] = F[read_2_return[key]][cond]
            elif key == "x":
                fofs[key] = x[cond]
            elif key == "y":
                fofs[key] = y[cond]
            elif key == "z":
                fofs[key] = z[cond]
            else:
                print("unrecognized key:", key)

        halo_star_ids = F["halo star ID"][()]

        cum_star_nb = np.cumsum(F["stellar count"])[cond]

    return (fofs, halo_star_ids, cum_star_nb)
