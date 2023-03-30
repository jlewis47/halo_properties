"""
TODO:Add individual stellar particle fescs as in other verstion
"""

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# import matplotlib.patches as pat
# from read_radgpu import o_rad_cube_big
# from read_stars import read_all_star_files
# from scipy.spatial import KDTree

# from tempfile import mkdtemp

# import time
# import string
import argparse
import os
from ..association.read_assoc_latest import read_assoc
from ..files.read_stars import read_specific_stars
from ..files.read_fullbox_big import *
from ..utils.utils import ll_to_fof_suffix, get_r200_suffix, get_suffix
from ..utils.functions_latest import *
from ..src.bpass_fcts import (
    get_mag_tab_BPASSV221_betas,
    get_mag_interp_fct,
    get_xis_interp_fct,
    get_star_mags_metals,
    get_star_xis_metals,
    bin_star_info,
    comp_betas,
)
from ..src.ray_fcts import sph_2_cart, cart_2_sph, sum_over_rays_bias, sum_over_rays_bias_nopython, sum_over_rays_bias_multid
from ..dust.dust_opacity import shoot_star_path_cheap, shoot_star_path, shoot_star_path_cheap_multid

import healpy as hp

# from dust_opacity import *
from ..files.wrap_boxes import *
from ..utils.output_paths import *
from mpi4py import MPI
from ..utils.units import get_unit_facts, convert_temp
from ..utils.utils import divide_task  # , sum_arrays_to_rank0
from ..params.params import *
# from time import sleep
from ..dust.att_coefs import att_coefs, att_coef_draine_file, get_dust_att_files

# from numba import jit

# from plot_functions import make_figure


def correct_lums_in_file(
   
    out_file,
    halo_Lintrs,

):


    with h5py.File(out_file, "a") as out_halos:
        
            # print(halo_Lintrs[:,0])
        for ibin, name in enumerate(Lintr_names):

            old_lintrs = out_halos[name]
            # print(out_file, old_lintrs.shape, halo_Lintrs[:,ibin].shape)
            old_lintrs[...] = halo_Lintrs[:,ibin] 

 

def correct_lintr(
    out_nb, overwrite=False, rtwo_fact=1, fesc_rad=1.0, ll=0.2, assoc_mthd="", test=False, dilate=8
):


    # fesc_debug_avg = 0
    # debug_counts = 0

    # small_f = 1e-15

    overstep = 1.2  # important !!! For overstep=1.2 instead of loading a subbox of say 512 cells per side
    # we add edges based on the repetittion/full extent of the simulattion so that we actaully work
    # with a box that has 512*1.2 cells per side.
    # this is done to handle haloes that are on the edges of sub boxes without exploding memory costs
    # unfortunately since we are bounded by memory (on buffy for example) this method leads to multiple
    # loads of the same subcube when processing the simulattion.

    assert fesc_rad >= 1.0, "fesc_rad must be larger or equal to one (fesc integration radius must include stellar association radius)"

    comm = MPI.COMM_WORLD
    Nproc = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:

        print("Running on %i tasks" % Nproc)

        if test:
            print("Dry run for testing purposes")
        elif overwrite:
            print("Overwriting existing output files")
        else:
            print("Skipping existing files")

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    frad_suffix = get_frad_suffix(fesc_rad)
    suffix = get_suffix(fof_suffix, rtwo_suffix, frad_suffix)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    if rank == 0 and not os.path.exists(analy_out):
        os.makedirs(analy_out)

    comm.Barrier()

    output_str = "output_%06i" % out_nb

    info_path = os.path.join(sim_path, output_str, "group_000001")
    snap_box_path = os.path.join(box_path, output_str)

    plt.rcParams.update({"font.size": 18})

    # find number of subcubes
    if rank==0:
        data_files = os.listdir(os.path.join(box_path, output_str))
        rho_files = [f for f in data_files if f[:3] == "rho" and "." not in f]

        n_subcubes = len(rho_files)

    else:

        n_subcubes = None

    n_subcubes = comm.bcast(n_subcubes, root=0)

    # print(rank, n_subcubes)


    assert (
        n_subcubes > 1
    ), "Couldn't find any 'rho' subcubes... Are you sure about the path?"
    # print(n_subcubes)
    subs_per_side = int(np.round(n_subcubes ** (1.0 / 3)))
    # print(subs_per_side)
    sub_side = int(float(ldx) / subs_per_side)
    # print(sub_side)

    # Get scale factor and co
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

    redshift = 1.0 / a - 1.0

    if rank == 0:
        print("Redshift is %.1f" % redshift)

    with open(os.path.join(out, "Mp"), "rb") as mass_file:
        Mp = np.fromfile(mass_file, dtype=np.float64)
    if rank == 0:
        print("DM part mass in msun : %e" % Mp)

    dist_obs = 345540.98618977674  # distance to obs point from box (0,0,0); in number of cells for z=6

    (
        mags,
        xis,
        contbetalow,
        contbetahigh,
        beta,
        metal_bins,
        age_bins,
    ) = get_mag_tab_BPASSV221_betas(bpass_file_name)
    
    xis_fct = get_xis_interp_fct(xis, age_bins, metal_bins)

    upper = 27
    # grid = np.mgrid[0:upper,0:upper,0:upper]/float(upper-1)

    if rank == 0:
        print("Getting halos and associated stars")

    # halo_tab = halo_tab[:5000]  # for testing

    # print(halo_tab)

    ran = [-ldx, 0, ldx]
    pos_vects = np.asarray([[i, j, k] for k in ran for j in ran for i in ran])



    fmin, fmax, f_per_proc = divide_task(n_subcubes, Nproc, rank)

    # print(rank, fmin, fmax)

    for z_subnb in range(subs_per_side):
        for y_subnb in range(subs_per_side):
            for x_subnb in range(subs_per_side):

                subcube_nb = np.ravel_multi_index(
                    (x_subnb, y_subnb, z_subnb),
                    (subs_per_side, subs_per_side, subs_per_side),
                )

                if test and subcube_nb >= Nproc:
                    continue

                if subcube_nb < fmin or subcube_nb >= fmax:
                    continue

                out_file = os.path.join(analy_out, "halo_stats_%i" % subcube_nb)

                # print(rank, subcube_nb)

                out_exists = os.path.exists(out_file)

                if not out_exists:

                    print(
                        "RANK %i: Couldn't find file for subcube #%i"
                        % (rank, subcube_nb)
                    )
                    continue

                sub_halo_tab, halo_star_ids, sub_halo_tot_star_nb = read_assoc(
                    out_nb,
                    "CoDaIII",
                    ldx,
                    sub_side,
                    rtwo_fact=rtwo_fact,
                    ll=ll,
                    assoc_mthd=assoc_mthd,
                    subnb=subcube_nb,
                )

                # print(len(sub_halo_tab), len(sub_halo_tot_star_nb))

                if len(sub_halo_tab) < 1:
                    continue

                loc_pos_nrmd = np.asarray(
                    [list(col) for col in sub_halo_tab[["x", "y", "z"]]]
                )
                loc_pos_nrmd = loc_pos_nrmd % ldx

                # coords_tree = KDTree(pos_nrmd, boxsize=ldx)

                # if rank == 0:
                #     print("Built halo KDTree")
                #     print("Reading subcube #%s" % (subcube_nb))

                # print(halo_tab["x"])

                # Retain halos within sub cube
                # x_cond = np.all(
                #     [
                #         halo_tab["x"] <= (x_subnb + 1) * sub_side,
                #         halo_tab["x"] > (x_subnb) * sub_side,
                #     ],
                #     axis=0,
                # )
                # y_cond = np.all(
                #     [
                #         halo_tab["y"] <= (y_subnb + 1) * sub_side,
                #         halo_tab["y"] > (y_subnb) * sub_side,
                #     ],
                #     axis=0,
                # )
                # z_cond = np.all(
                #     [
                #         halo_tab["z"] <= (z_subnb + 1) * sub_side,
                #         halo_tab["z"] > (z_subnb) * sub_side,
                #     ],
                #     axis=0,
                # )

                # ind_subcube = x_cond * y_cond * z_cond

                # print(np.sum(ind_subcube))

                # sub_halo_tab = halo_tab.compress(ind_subcube, axis=0)
                # sub_halo_tot_star_nb = tot_star_nbs.compress(ind_subcube)
                sub_halo_star_nb = sub_halo_tab["nstar"]
                sub_idxs = sub_halo_tab["ids"]

                limit_r = sub_halo_tab["rpx"] + 1
                sample_r = do_half_round(limit_r)

                pos = do_half_round(loc_pos_nrmd)  # was np.int16

                # (0,0,0) px locattion of sub_side**3 cube within whole data set
                edge = np.asarray(
                    [x_subnb * sub_side, y_subnb * sub_side, z_subnb * sub_side]
                )

                ctr_bxd = pos - edge

                lower_bounds = np.int32(ctr_bxd - sample_r[:, np.newaxis])
                upper_bounds = np.int32(ctr_bxd + sample_r[:, np.newaxis])

                halo_Lintrs = np.zeros(
                    (np.shape(sub_halo_tab)[0], len(sfr_age_bins)), dtype=np.float64
                )
                halo_SFRs = np.zeros(
                    (np.shape(sub_halo_tab)[0], len(sfr_age_bins)), dtype=np.float32
                )


                if rank == 0:
                    print("Allocated data arrays")


                edge_overstep = int(sub_side * (overstep - 1) * 0.5)

                ctr_bxd = ctr_bxd + edge_overstep
                lower_bounds = lower_bounds + edge_overstep
                upper_bounds = upper_bounds + edge_overstep

 

                for ind, halo in enumerate(sub_halo_tab):

                    # print('    Halo #%i'%ind)

                    r_px = halo["rpx"]



                    # If there aren't any stars : no need to calculate emissivities or star formattion stuff
                    if sub_halo_star_nb[ind] > 0:

                        # Get stars for halo  from list of associated stars
                        cur_star_ids = halo_star_ids[
                            sub_halo_tot_star_nb[ind]
                            - sub_halo_star_nb[ind] : sub_halo_tot_star_nb[ind]
                        ]


                        # print(cur_star_ids)

                        # print(sub_halo_tot_star_nb[ind] - sub_halo_star_nb[ind], sub_halo_tot_star_nb[ind])

                        cur_stars = read_specific_stars(
                            os.path.join(star_path, output_str), cur_star_ids
                        )

                        cur_star_luminosity = (
                            10
                            ** (
                                get_star_xis_metals(
                                    cur_stars["age"],
                                    cur_stars["Z/0.02"] * 0.02,
                                    xis_fct,
                                )
                            ) * cur_stars['mass'] / (1.0 - eta_sn)
                        )  # ph/s



                        halo_SFRs[ind], halo_Lintrs[ind] = bin_star_info(
                            halo_SFRs[ind],
                            halo_Lintrs[ind],
                            cur_stars,
                            cur_star_luminosity,
                            sfr_age_bins,
                        )


                    # if test and halo["mstar"] > 0:
                    if test and halo["mass"] * Mp > 1.0e11:
                        print("found a big halo")
                        print("%E" % (halo["mass"] * Mp))
                        print("Lintr=%E ph/s" % halo_Lintrs[ind])


            # print(halo_2e4k_gmass, halo_2e4k_gmass.max(), np.mean(halo_2e4k_gmass))

                if not test:
                    pos = np.transpose([sub_halo_tab[key] for key in ["x", "y", "z"]])

                    print("Writing %s" % out_file)

        

                    correct_lums_in_file(
                        out_file,
                        halo_Lintrs,

                    )

                else:

                    print('Test run, out_file: %s'%out_file)

                    return(sub_halo_tab["mass"] * Mp, halo_Lintrs)

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
        default=1
    )

    Arg_parser.add_argument(
        "--ll", metavar="ll", type=float, help="linking length for fof", default=0.2
    )
    Arg_parser.add_argument(
        "--assoc_mthd",
        metavar="assoc_mthd",
        type=str,
        help="method for linking stars to fof",
        default=""
    )
    Arg_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="When used, code overwrites all found data",
        default=False
    )
    Arg_parser.add_argument(
        "--test",
        action="store_true",
        help="When used, code runs on one subcube and doesn't write",
        default=False
    )
    Arg_parser.add_argument(
        "--dilate",
        type=int,
        help="number of times to resample grid when performing sums within r_px",
        default=8
    )

    args = Arg_parser.parse_args()

    out_nb = args.nb
    rtwo_fact = args.rtwo_fact
    assoc_mthd = args.assoc_mthd
    ll = args.ll
    overwrite = args.overwrite
    dilate = args.dilate
    fesc_rad = args.fesc_rad

    correct_lintr(
        out_nb,
        rtwo_fact=rtwo_fact,
        fesc_rad=fesc_rad,
        assoc_mthd=assoc_mthd,
        ll=ll,
        overwrite=overwrite,
        test=args.test,
        dilate=dilate
    )
