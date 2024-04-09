"""
------
Read halo and star files and make associations
Saved as binary file that repeats for N halos and another that contains M stars
Associated stars and non associated stars are saved to separate binary files
Two files track each halo star number and 1st star file position
A final file saves halo IDs
------

This verstion includes halos that are close to each other and/or without stars
It also tracks stars that don't end up associated, that are saved to separate 'LONE' star files

"""


from halo_properties.params.params import *

import numpy as np

# import matplotlib as mpl

# mpl.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.patches as pat


from halo_properties.files.read_stars import read_rank_star_files

# from scipy import spatial
# import times

# import string
import argparse
import os
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import *
from halo_properties.utils.utils import *

# from halo_properties.association.read_fof import o_fof, o_luke_fof, o_mp_fof
from halo_properties.association.read_assoc_latest import read_assoc

# from read_fof import o_fof
import h5py
from halo_properties.utils.output_paths import *
from halo_properties.files.wrap_boxes import *

# from scipy.stats import binned_statistic_dd
import h5py
from mpi4py import MPI


def find_lone_stars(
    out_nb,
    rtwo_fact=1,
    npart_thresh=50,
    assoc_mthd="",
    overwrite=False,
    binary_fof=False,
    mp_fof=False,
    clean=False,
    ll=0.2,
):
    check_assoc_keys(assoc_mthd)

    # phew_path='/data2/jlewis/dusts/output_00'+out_nb
    # star_path='/data2/jlewis/dusts/'
    # info_path='/data2/jlewis/dusts/output_00'+out_nb

    if sixdigits:
        output_str = "output_%06i" % out_nb
    else:
        output_str = "output_%05i" % out_nb

    info_path = os.path.join(sim_path, output_str)
    if sixdigits:
        info_path = os.path.join(info_path, "group_000001")

    loc_star_path = os.path.join(star_path, output_str)

    dset = dataset(rtwo=rtwo_fact, ll=ll, assoc_mthd=assoc_mthd, clean=clean, mp=mp_fof)

    out, assoc_out, analy_out, suffix = gen_paths(sim_name, out_nb, dset)

    Np_tot = ldx**3

    """Get scale factor and co"""
    (
        t,
        a,
        H0,
        om_m,
        om_l,
        om_k,
        om_b,
        unit_l,
        unit_d,
        unit_t,
        l,
        Lco,
        L,
        px_to_m,
    ) = get_infos(info_path, out_nb, ldx)

    comm = MPI.COMM_WORLD
    Nproc = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print("Running on snapshot %s" % output_str)
    if rank == 0:
        print("using r=%iXr200" % rtwo_fact)

    tstep = 0.0  # temporary

    out_file = os.path.join(assoc_out, ("lone_stars_%s" % out_nb) + suffix)
    if os.path.exists(out_file) and not overwrite:
        if rank == 0:
            print("File %s exists and I'm not in overwrite mode ... Exiting" % out_file)
        return ()

    Mp = get_Mp(om_m, om_b, H0, Lco, Np_tot)
    if rank == 0:
        print("Particle mass is :", Mp)

    if not "Mp" in os.listdir(out):
        out_Mp = open(os.path.join(out, "Mp"), "wb")
        out_Mp.write(np.float64(Mp))
        out_Mp.close()

    if rank == 0:
        print("Reading association")

    # read_association result
    (
        _,
        halo_star_ids,
        _,
    ) = read_assoc(out_nb, sim_name, dset, return_keys=[])
    # ) = read_assoc(out_nb, sim_name, dset, return_keys=["ids", "mstar", "nstar"])

    # halo_halo_ids = fof_cat["ids"]
    # halo_stellar_mass = fof_cat["mstar"]
    # halo_star_nb = fof_cat["nstar"]

    set_assoc_stars = set(halo_star_ids)

    if rank == 0:
        print("Reading stars")

    # Stars
    time_myr, stars, star_ids = read_rank_star_files(loc_star_path, rank, Nproc)

    families = stars["family"]
    stars = stars[families == 2]
    star_ids = star_ids[families == 2]

    stars = np.transpose(
        [
            stars["mass"],
            stars["x"],
            stars["y"],
            stars["z"],
            stars["age"],
            stars["Z/0.02"],
        ]
    )
    # filter non-stars (->no debris)
    set_star_ids = set(star_ids)

    unassoc_ids = list(set_star_ids - set_assoc_stars)
    unassoc_stars = stars[np.asarray(unassoc_ids) == star_ids, :]

    unassoc_x = unassoc_stars[:, 1]
    unassoc_y = unassoc_stars[:, 2]
    unassoc_z = unassoc_stars[:, 3]
    unassoc_mass = unassoc_stars[:, 0]
    unassoc_age = unassoc_stars[:, 4]
    unassoc_Z = unassoc_stars[:, 5]

    comm.Barrier()

    # halo_star_nb = sum_arrays_to_rank0(comm, np.ascontiguousarray(halo_star_nb))
    # halo_star_ids = merge_arrays_rank0(comm, np.concatenate(halo_star_ids))
    # halo_halo_ids = merge_arrays_rank0(comm, np.concatenate(halo_halo_ids))
    # halo_stellar_mass = sum_arrays_to_rank0(comm, halo_stellar_mass)
    # norm = sum_arrays_to_rank0(comm, np.int8(np.all(new_ctrs != 0, axis=1)))
    # new_ctrs = sum_arrays_to_rank0(comm, new_ctrs)  #

    print(
        unassoc_ids,
        unassoc_x,
        unassoc_y,
        unassoc_z,
        unassoc_mass,
        unassoc_age,
        unassoc_Z,
    )

    tot_unassoc_ids = merge_arrays_rank0(comm, unassoc_ids, dtype=np.int64)

    tot_unassoc_x = merge_arrays_rank0(comm, unassoc_x, dtype=np.float64)
    tot_unassoc_y = merge_arrays_rank0(comm, unassoc_y, dtype=np.float64)
    tot_unassoc_z = merge_arrays_rank0(comm, unassoc_z, dtype=np.float64)

    tot_unassoc_mass = merge_arrays_rank0(comm, unassoc_mass, dtype=np.float32)
    tot_unassoc_age = merge_arrays_rank0(comm, unassoc_age, dtype=np.float32)
    tot_unassoc_Z = merge_arrays_rank0(comm, unassoc_Z, dtype=np.float32)

    comm.Barrier()

    if rank == 0:
        print("Found lone stars")

        unassoc_ids, idx = np.unique(tot_unassoc_ids, return_index=True)

        unassoc_x = tot_unassoc_x[idx]
        unassoc_y = tot_unassoc_y[idx]
        unassoc_z = tot_unassoc_z[idx]

        unassoc_mass = tot_unassoc_mass[idx]
        unassoc_age = tot_unassoc_age[idx]
        unassoc_Z = tot_unassoc_Z[idx]

        with h5py.File((out_file), "w") as out_halos:
            out_halos.create_dataset(
                "unassoc_ids", data=unassoc_ids, dtype=np.int64, compression="lzf"
            )

            out_halos.create_dataset(
                "unassoc_x", data=unassoc_x, dtype=np.float64, compression="lzf"
            )
            out_halos.create_dataset(
                "unassoc_y", data=unassoc_y, dtype=np.float64, compression="lzf"
            )
            out_halos.create_dataset(
                "unassoc_z", data=unassoc_z, dtype=np.float64, compression="lzf"
            )

            out_halos.create_dataset(
                "unassoc_mass", data=unassoc_mass, dtype=np.float32, compression="lzf"
            )
            out_halos.create_dataset(
                "unassoc_age", data=unassoc_age, dtype=np.float32, compression="lzf"
            )
            out_halos.create_dataset(
                "unassoc_Z", data=unassoc_Z, dtype=np.float32, compression="lzf"
            )

            print("Done")


"""
Main body
"""


if __name__ == "__main__":
    Arg_parser = argparse.ArgumentParser("Associate stars and halos in full simulation")

    Arg_parser.add_argument(
        "nb",
        metavar="nsnap",
        type=int,
        help='snap number string, give as "XXX" so that 00001 is "001" and 00111 is "111"',
    )

    Arg_parser.add_argument(
        "--rtwo_fact",
        metavar="rtwo_fact",
        type=float,
        help="1.0 -> associate stellar particles within 1xR200 for all haloes",
        default=1,
    )
    Arg_parser.add_argument(
        "--npart_thresh",
        metavar="npart_thresh",
        type=float,
        help="dark matter particle number threshold for halos",
        default=50,
    )
    Arg_parser.add_argument(
        "--ll",
        metavar="fof linking length",
        type=float,
        help="dark matter particle number threshold for halos",
        default=0.2,
    )
    Arg_parser.add_argument(
        "--assoc_mthd",
        metavar="assoc_mthd",
        type=str,
        help="method for linking stars to fof",
        default="",
    )
    Arg_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing files?",
        default=False,
    )
    Arg_parser.add_argument(
        "--binary_fof",
        action="store_true",
        help="fof in binary format?",
        default=False,
    )
    Arg_parser.add_argument(
        "--mp",
        action="store_true",
        help="fof with watershed segmentation by Mei Palanque",
        default=False,
    )
    Arg_parser.add_argument(
        "--clean",
        action="store_true",
        help="use cleaned catalogue",
        default=False,
    )

    args = Arg_parser.parse_args()

    out_nb = args.nb
    rtwo_fact = args.rtwo_fact
    npart_thresh = args.npart_thresh
    assoc_mthd = args.assoc_mthd

    assert not (
        args.binary_fof and args.mp
    ), "Can't have both binary and mp segmentation"

    find_lone_stars(
        out_nb,
        npart_thresh=npart_thresh,
        rtwo_fact=rtwo_fact,
        assoc_mthd=assoc_mthd,
        overwrite=args.overwrite,
        binary_fof=args.binary_fof,
        mp_fof=args.mp,
        clean=args.clean,
        ll=0.2,
    )
