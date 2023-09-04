
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.spatial import KDTree
import argparse

from mpi4py import MPI

from halo_properties.params.params import *
from halo_properties.association.read_assoc_latest import read_assoc
from halo_properties.utils.utils import get_fof_suffix, get_r200_suffix, get_suffix, ll_to_fof_suffix
from halo_properties.utils.output_paths import gen_paths
from halo_properties.files.read_stars import read_specific_stars
from halo_properties.utils.utils import merge_arrays_rank0, sum_arrays_to_rank0

def clean_assoc(out_nb,
        rtwo_fact=1.0,
        assoc_mthd="stellar_peak",
        ll=0.2,
        overwrite=False,
        mp=True):


    if sixdigits:
        output_str = "output_%06i" % out_nb
    else:
        output_str = "output_%05i" % out_nb

    info_path = os.path.join(sim_path, output_str)
    if sixdigits:
        info_path = os.path.join(info_path, "group_000001")

    loc_star_path = os.path.join(star_path, output_str)
    # phew_path=os.path.join(path,output_str)
    if "ll" in fof_path:
        fof_suffix = get_fof_suffix(fof_path)
    else:
        fof_suffix = ll_to_fof_suffix(ll)

    # print(fof_path, fof_suffix)

    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix=fof_suffix, rtwo_suffix=rtwo_suffix, mp=mp)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)
    out_file = os.path.join(assoc_out, ("assoc_halos_clean_%s" % out_nb) + suffix)

    if os.path.isfile(out_file) and not overwrite:
        if rank==0 : print(f"File exists ({out_file:s}) and overwrite is False")
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank==0:
        print("Reading existing association")

    sub_halo_tab, halo_star_ids, sub_halo_tot_star_nb = read_assoc(
        out_nb,
        sim_name,
        ldx,
        512,
        rtwo_fact=rtwo_fact,
        ll=ll,
        assoc_mthd=assoc_mthd,
        mp=mp,
        clean=False
    )

    uniques, u_inverse, u_count = np.unique(halo_star_ids, return_counts=True, return_inverse=True)

    mult_uniques = uniques[u_count > 1][:]

    loc_mult_uniques = np.array_split(mult_uniques, size)[rank]
    loc_mult_u_counts = np.array_split(u_count[u_count > 1][:], size)[rank]


    if rank==0:
        print(f"Found {len(mult_uniques)} stars in multiple halos... processing")

    halo_nstar_red = np.zeros_like(sub_halo_tab["nstar"], dtype=np.int32)
    halo_mstar_red = np.zeros_like(sub_halo_tab["nstar"], dtype=np.float32)
    stellar_id_args_to_del = np.zeros(np.sum(loc_mult_u_counts)-len(loc_mult_u_counts), dtype=np.int32)

    sub_halo_tot_star_nb_m1 = sub_halo_tot_star_nb - sub_halo_tab["nstar"]

    i_id_to_del = 0

    for itgt,tgt_unique in enumerate(loc_mult_uniques):

        if rank==0 and itgt%1000==0:
            print(f"rank 0 is processing {itgt}/{len(loc_mult_uniques)}")

        star_u_args = (halo_star_ids == tgt_unique).nonzero()[0]
        # u_halo_args = np.digitize(star_u_args, sub_halo_tot_star_nb)
        u_halo_args = np.searchsorted(sub_halo_tot_star_nb, star_u_args, side="right")
        # print(star_u_args)
        # print(np.searchsorted(sub_halo_tot_star_nb, star_u_args), u_halo_args)

        u_halo_coords = np.array(sub_halo_tab[u_halo_args][["x","y","z"]].tolist())

        star = read_specific_stars(os.path.join(star_path, f"output_{out_nb:06d}"), np.asarray([tgt_unique]))
        star_pos = np.asarray(star[["x","y","z"]].tolist())*ldx

        dists = np.linalg.norm(u_halo_coords - star_pos, axis=1)
        # print(dists)
        not_closest_u_halo = u_halo_args[dists != np.min(dists)]

        halo_nstar_red[not_closest_u_halo] = halo_nstar_red[not_closest_u_halo] - 1
        halo_mstar_red[not_closest_u_halo] = halo_mstar_red[not_closest_u_halo] - star["mass"][0]

        for h_arg in not_closest_u_halo:

            lo,hi = sub_halo_tot_star_nb_m1[h_arg], sub_halo_tot_star_nb[h_arg]

            # if rank==0:print(stellar_id_args_to_del, i_id_to_del, star_u_args[(star_u_args<hi) * (star_u_args>=lo)], lo, hi)

            stellar_id_args_to_del[i_id_to_del] = (star_u_args[(star_u_args<hi) * (star_u_args>=lo)])[0]
            i_id_to_del+=1

    stellar_id_args_to_del = stellar_id_args_to_del[stellar_id_args_to_del>0]

    comm.Barrier()

    if rank==0:
        print("Cleaning targets found merging target on rank 0")

    # print(rank, stellar_id_args_to_del, halo_nstar_red)

    #gather all the stellar_id_args_to_del and halo_nstar_red
    stellar_id_args_to_del = merge_arrays_rank0(comm, np.int32(stellar_id_args_to_del), dtype=np.int32)

    halo_nstar_red = sum_arrays_to_rank0(comm, np.int32(halo_nstar_red))
    halo_mstar_red = sum_arrays_to_rank0(comm, np.float32(halo_mstar_red))


    comm.Barrier()

    if rank==0:

        print("rank 0 is cleaning targets...")

        # print(len(halo_star_ids), len(halo_nstar_red), sub_halo_tab["nstar"].shape)

        # stellar_id_args_to_del = np.concatenate(stellar_id_args_to_del)
        # halo_mstar_red = np.sum(halo_mstar_redaxis = 0)
        # halo_nstar_red = np.sum(halo_nstar_redaxis = 0)

        # print(len(halo_star_ids), halo_nstar_red.shape, sub_halo_tab["nstar"].shape)
        # print(0, stellar_id_args_to_del, halo_nstar_red)



        updated_halo_star_ids = np.delete(halo_star_ids, stellar_id_args_to_del)
        updated_sub_halo_nstar = sub_halo_tab["nstar"] + halo_nstar_red
        updated_sub_halo_mstar = sub_halo_tab["mstar"] + halo_mstar_red


        print("rank 0 is saving new tables...")

        with h5py.File((out_file), "w") as out_halos:
            out_halos.create_dataset(
                "ID", data=np.int64(sub_halo_tab["ids"][()]), dtype=np.int64, compression="lzf"
            )
            # out_halos.create_dataset(
            #     "fnb", data=halo_fnbs, dtype=np.int32, compression="lzf"
            # )
            # out_halos.create_dataset(
            #     "lnb", data=halo_lnbs, dtype=np.int32, compression="lzf"
            # )
            out_halos.create_dataset(
                "mass", data=sub_halo_tab["mass"][()], dtype=np.float32, compression="lzf"
            )
            out_halos.create_dataset(
                "coords", data=np.asarray(sub_halo_tab[["x","y","z"]].tolist()), dtype=np.float64, compression="lzf"
            )
            out_halos.create_dataset(
                "rpx", data=sub_halo_tab["rpx"][()], dtype=np.float32, compression="lzf"
            )

            out_halos.create_dataset(
                "stellar mass",
                data=updated_sub_halo_mstar,
                dtype=np.float32,
                compression="lzf",
            )
            out_halos.create_dataset(
                "stellar count", data=updated_sub_halo_nstar, dtype=np.int32, compression="lzf"
            )
            out_halos.create_dataset(
                "halo star ID",
                data=np.int64(updated_halo_star_ids),
                dtype=np.int64,
                compression="lzf",
            )

            print("Done")



        # print(len(updated_halo_star_ids), len(updated_sub_halo_nstar))
        # print(np.sum(updated_sub_halo_nstar), np.sum(sub_halo_tab["nstar"]))



"""
Main body
"""


if __name__ == "__main__":
    Arg_parser = argparse.ArgumentParser("Compute gas and stellar properties in halos")

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
        "--ll", metavar="ll", type=float, help="linking length for fof", default=0.2
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
        help="When used, code overwrites all found data",
        default=False,
    )

    Arg_parser.add_argument(
        "--mp",
        # metavar="mp_segmentation",
        action="store_true",
        help="Use Mei Palanque's watershed segmentation catalogue",
        default=False,
    )

    args = Arg_parser.parse_args()

    out_nb = args.nb
    rtwo_fact = args.rtwo_fact
    assoc_mthd = args.assoc_mthd
    ll = args.ll
    overwrite = args.overwrite
    mp = args.mp


    clean_assoc(
        out_nb,
        rtwo_fact=rtwo_fact,
        assoc_mthd=assoc_mthd,
        ll=ll,
        overwrite=overwrite,
        mp=mp,
    )
