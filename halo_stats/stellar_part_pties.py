"""
TODO:Add individual stellar particle fescs as in other verstion
"""

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import pandas as pd

# import matplotlib.patches as pat
# from read_radgpu import o_rad_cube_big
# from read_stars import read_all_star_files
# from scipy.spatial import KDTree

# from tempfile import mkdtemp

# import time
# import string
import argparse
import os
from scipy.stats import binned_statistic
from halo_properties.association.read_assoc_latest import read_assoc
from halo_properties.files.read_stars import read_specific_stars
from halo_properties.files.read_fullbox_big import *
from halo_properties.utils.utils import ll_to_fof_suffix, get_r200_suffix, get_suffix
from halo_properties.utils.functions_latest import *
from halo_properties.src.bpass_fcts import (
    get_mag_tab_BPASSV221_betas,
    get_mag_interp_fct,
    get_xis_interp_fct,
    get_star_mags_metals,
    get_star_xis_metals,
    comp_betas_indv,
)
from halo_properties.src.ray_fcts import (
    sph_2_cart,
    cart_2_sph,
    sum_over_rays_bias,
    sum_over_rays_bias_nopython,
    sum_over_rays_bias_multid,
)
from halo_properties.dust.dust_opacity import (
    shoot_star_path_cheap,
    shoot_star_path,
    shoot_star_path_cheap_multid,
)

import healpy as hp

# from dust_opacity import *
from halo_properties.files.wrap_boxes import *
from halo_properties.utils.output_paths import *
from mpi4py import MPI
from halo_properties.utils.units import get_unit_facts, convert_temp
from halo_properties.utils.utils import divide_task  # , sum_arrays_to_rank0
from halo_properties.params.params import *

# from time import sleep
from halo_properties.dust.att_coefs import (
    att_coefs,
    att_coef_draine_file,
    get_dust_att_files,
)

# from numba import jit

# from plot_functions import make_figure


def write_fields(
    out_file,
    rho,
    rhod,
    xion,
    temp,
    age,
    Z,
    mag,
    beta,
    fesc,
    mass,
    lintr,
    radius,
    att_sets,
):
    nsets = len(att_sets)

    with h5py.File(out_file, "w") as out_halos:
        dset = out_halos.create_dataset(
            "gas density", data=rho, dtype="f4", compression="lzf"
        )
        dset.attrs["unit"] = "g.cm^-3"

        dset = out_halos.create_dataset(
            "dust density", data=rhod, dtype="f4", compression="lzf"
        )
        dset.attrs["unit"] = "g.cm^-3"

        dset = out_halos.create_dataset(
            "gas xion", data=xion, dtype="f4", compression="lzf"
        )
        dset.attrs["unit"] = ""
        dset.attrs["descrption"] = "fraction of ionised gas"

        dset = out_halos.create_dataset(
            "gas temperature", data=temp, dtype="f4", compression="lzf"
        )
        dset.attrs["unit"] = "K"

        dset = out_halos.create_dataset(
            "stellar mass", data=mass, dtype="f4", compression="lzf"
        )
        dset.attrs["unit"] = "Msun"

        dset = out_halos.create_dataset(
            "halo lintr", data=lintr, dtype="f8", compression="lzf"
        )
        dset.attrs["unit"] = "s^-1"

        dset = out_halos.create_dataset(
            "stellar age", data=age, dtype="f4", compression="lzf"
        )
        dset.attrs["unit"] = "Myr"

        dset = out_halos.create_dataset(
            "stellar Z", data=Z, dtype="f4", compression="lzf"
        )
        dset.attrs["unit"] = "absolute metallicity"

        dset = out_halos.create_dataset(
            "stellar radii", data=radius, dtype="f4", compression="lzf"
        )
        dset.attrs["unit"] = "cells"

        for iset in range(nsets):
            dset = out_halos.create_dataset(
                "stellar mag %s" % att_sets[iset].name,
                data=mag[iset],
                dtype="f4",
                compression="lzf",
            )
            dset.attrs["unit"] = ""
            dset.attrs[
                "description"
            ] = "AB magnitude at 1600 Ang (extinction 1D, LoS direction = 0)"

            dset = out_halos.create_dataset(
                "stellar beta %s" % att_sets[iset].name,
                data=beta[iset],
                dtype="f4",
                compression="lzf",
            )
            dset.attrs["unit"] = ""
            dset.attrs["description"] = "UV slope (reddening 1D, LoS direction = 0)"

            dset = out_halos.create_dataset(
                "stellar fesc %s" % att_sets[iset].name,
                data=fesc[iset],
                dtype="f8",
                compression="lzf",
            )
            dset.attrs["unit"] = "g.cm^-3"
            dset.attrs[
                "description"
            ] = "ray-traced fraction of escaping photons that reach IGM"


def compute_part_pties(
    out_nb,
    overwrite=False,
    rtwo_fact=1,
    ll=0.2,
    assoc_mthd="",
    test=False,
    mbin=1e11,
    mbin_width=1e10,
    mp=False,
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
    frad_suffix = get_frad_suffix(1.0)
    suffix = get_suffix(fof_suffix, rtwo_suffix, frad_suffix, mp=mp)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    if rank == 0 and not os.path.exists(analy_out):
        os.makedirs(analy_out)

    comm.Barrier()

    output_str = "output_%06i" % out_nb

    info_path = os.path.join(sim_path, output_str, "group_000001")
    snap_box_path = os.path.join(box_path, output_str)

    plt.rcParams.update({"font.size": 18})

    # find number of subcubes
    if rank == 0:
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
    mags_fct = get_mag_interp_fct(mags, age_bins, metal_bins)
    low_mags_fct = get_mag_interp_fct(contbetalow, age_bins, metal_bins)
    high_mags_fct = get_mag_interp_fct(contbetahigh, age_bins, metal_bins)
    xis_fct = get_xis_interp_fct(xis, age_bins, metal_bins)

    upper = 27
    # grid = np.mgrid[0:upper,0:upper,0:upper]/float(upper-1)

    if rank == 0:
        print("Getting halos and associated stars")

    # halo_tab = halo_tab[:5000]  # for testing

    # print(halo_tab)

    ran = [-ldx, 0, ldx]
    pos_vects = np.asarray([[i, j, k] for k in ran for j in ran for i in ran])

    rho_fact = get_unit_facts("rho", px_to_m, unit_d, unit_l, unit_t, a)
    rhod_fact = get_unit_facts("rhod", px_to_m, unit_d, unit_l, unit_t, a)
    tau_fact = get_unit_facts("tau", px_to_m, unit_d, unit_l, unit_t, a)
    temp_fact = get_unit_facts("temp", px_to_m, unit_d, unit_l, unit_t, a)

    npix = hp.nside2npix(nside)
    pix = np.arange(npix)
    thes, phis = hp.pix2ang(nside, pix)
    # Xvects,Yvects,Zvects=hp.pix2vec(nside,pix) #with radius = 1

    # hp_surf_elem=4.*np.pi*px_to_m**2 #in m^2.dx^-2
    # nbs_tot=np.zeros((len(xbins)-1,len(ybins)-1))

    # setup dust attenuattion coefficient sets
    # first set corresponds to no dust attenuattion/extincttion !
    att_sets = [att_coefs("no_dust", 0.0, 0.0, 0.0, sixteen=0.0)]

    draine_files = get_dust_att_files()
    for f in draine_files:
        # print(f)
        att_sets.append(att_coef_draine_file(f))

    if rank == 0:
        print("I found %i sets of dust absorption coefficients" % (len(att_sets) - 1))

    assert (
        att_sets[0].Kappa912 == 0.0
    ), "First attenuattion coeff set must be null -> no case dust case"

    big_side = int(sub_side * overstep)

    big_rho = np.zeros((big_side, big_side, big_side), dtype=np.float32)
    big_rhod = np.zeros((big_side, big_side, big_side), dtype=np.float32)
    big_metals = np.zeros((big_side, big_side, big_side), dtype=np.float32)
    big_xion = np.zeros((big_side, big_side, big_side), dtype=np.float32)
    big_temp = np.zeros((big_side, big_side, big_side), dtype=np.float32)

    fmin, fmax, f_per_proc = divide_task(n_subcubes, Nproc, rank)

    # print(rank, fmin, fmax)

    for z_subnb in range(subs_per_side):
        for y_subnb in range(subs_per_side):
            for x_subnb in range(subs_per_side):
                subcube_nb = np.ravel_multi_index(
                    (x_subnb, y_subnb, z_subnb),
                    (subs_per_side, subs_per_side, subs_per_side),
                )

                if test and subcube_nb != 12:
                    continue

                if subcube_nb < fmin or subcube_nb >= fmax:
                    continue

                mbinstr = f"{mbin:.1e}".replace(".", "p")
                out_file = os.path.join(
                    analy_out, f"stellar_pties_mbin{mbinstr:s}_{subcube_nb:d}"
                )

                # print(rank, subcube_nb)

                out_exists = os.path.exists(out_file)

                if out_exists and not (overwrite or test):
                    print(
                        "RANK %i: Skipping subcube #%i since it already exists"
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
                    mp=mp,
                )

                # print(mbin-mbin_width, (sub_halo_tab["mass"] * Mp), mbin+mbin_width)

                # keep haloes in bin
                filt = ((mbin - mbin_width) < (sub_halo_tab["mass"] * Mp)) * (
                    (sub_halo_tab["mass"] * Mp) < (mbin + mbin_width)
                )
                sub_halo_tab = sub_halo_tab[filt]
                sub_halo_tot_star_nb = sub_halo_tot_star_nb[filt]

                # print(len(sub_halo_tab))

                if len(sub_halo_tab) < 1:
                    continue

                loc_pos_nrmd = np.asarray(
                    [list(col) for col in sub_halo_tab[["x", "y", "z"]]]
                )
                loc_pos_nrmd = loc_pos_nrmd % ldx

                sub_halo_star_nb = sub_halo_tab["nstar"]
                sub_idxs = sub_halo_tab["ids"]

                limit_r = sub_halo_tab["rpx"] + 1
                sample_r = do_half_round(limit_r)

                limit_r_fesc = sub_halo_tab["rpx"] + 1
                sample_r_fesc = do_half_round(limit_r_fesc)

                nb_stars = np.sum(sub_halo_star_nb)

                pos = do_half_round(loc_pos_nrmd)  # was np.int16

                # (0,0,0) px locattion of sub_side**3 cube within whole data set
                edge = np.asarray(
                    [x_subnb * sub_side, y_subnb * sub_side, z_subnb * sub_side]
                )

                ctr_bxd = pos - edge

                lower_bounds = np.int32(ctr_bxd - sample_r[:, np.newaxis])
                upper_bounds = np.int32(ctr_bxd + sample_r[:, np.newaxis])

                lower_bounds_fesc = np.int32(ctr_bxd - sample_r_fesc[:, np.newaxis])
                upper_bounds_fesc = np.int32(ctr_bxd + sample_r_fesc[:, np.newaxis])

                under = lower_bounds_fesc < 0
                over = upper_bounds_fesc > sub_side

                outside = [under, over]

                if rank == 0:
                    print("Allocated data arrays")

                # #profiles
                # halo_prof_rho = np.zeros((np.shape(sub_halo_tab)[0], len(profile_radius_bins)),dtype=np.float32)
                # halo_prof_Z = np.zeros((np.shape(sub_halo_tab)[0], len(profile_radius_bins)),dtype=np.float32)
                # halo_prof_temp = np.zeros((np.shape(sub_halo_tab)[0], len(profile_radius_bins)),dtype=np.float32)
                # halo_prof_xhi = np.zeros((np.shape(sub_halo_tab)[0], len(profile_radius_bins)),dtype=np.float32)
                # halo_prof_Mst = np.zeros((np.shape(sub_halo_tab)[0], len(profile_radius_bins)),dtype=np.float32)
                # halo_prof_dust = np.zeros((np.shape(sub_halo_tab)[0], len(profile_radius_bins)),dtype=np.float32)

                edge_overstep = int(sub_side * (overstep - 1) * 0.5)

                ctr_bxd = ctr_bxd + edge_overstep
                lower_bounds = lower_bounds + edge_overstep
                upper_bounds = upper_bounds + edge_overstep
                lower_bounds_fesc = lower_bounds_fesc + edge_overstep
                upper_bounds_fesc = upper_bounds_fesc + edge_overstep

                # import datas and add margin for edge cases !
                # with mpi we cant read the same files at the same time
                # smart people would avoid this, here we catch the error and make the task sleep
                # done = False
                # print(snap_box_path, outside, n_subcubes, sub_side, overstep)
                # while not done:
                # try:
                new_get_overstep_hydro_cubed(
                    big_rho,
                    subcube_nb,
                    snap_box_path,
                    "rho",
                    outside,
                    n_subcubes=n_subcubes,
                    size=sub_side,
                    overstep=overstep,
                    debug=test,
                )
                # done = True
                # except IOError:
                # sleep(1)

                # done = False
                # while not done:
                # try:
                new_get_overstep_hydro_cubed(
                    big_xion,
                    subcube_nb,
                    snap_box_path,
                    "xion",
                    outside,
                    n_subcubes=n_subcubes,
                    size=sub_side,
                    overstep=overstep,
                    debug=test,
                )
                # done = True
                # except IOError:
                # sleep(1)

                # done = False
                # while not done:
                # try:
                new_get_overstep_hydro_cubed(
                    big_metals,
                    subcube_nb,
                    snap_box_path,
                    "Z",
                    outside,
                    n_subcubes=n_subcubes,
                    size=sub_side,
                    overstep=overstep,
                    debug=test,
                )
                # done = True
                # except IOError:
                # sleep(1)

                # done = False
                # while not done:
                # try:
                new_get_overstep_hydro_cubed(
                    big_rhod,
                    subcube_nb,
                    snap_box_path,
                    "dust",
                    outside,
                    n_subcubes=n_subcubes,
                    size=sub_side,
                    overstep=overstep,
                    debug=test,
                )
                # done = True
                # except IOError:
                # sleep(1)

                # done = False
                # while not done:
                # try:
                new_get_overstep_hydro_cubed(
                    big_temp,
                    subcube_nb,
                    snap_box_path,
                    "temp",
                    outside,
                    n_subcubes=n_subcubes,
                    size=sub_side,
                    overstep=overstep,
                    debug=test,
                )
                # done = True
                # except IOError:
                # sleep(1)

                stellar_rho = np.zeros(nb_stars, dtype="f4")
                stellar_taus = np.zeros(nb_stars, dtype="f4")
                stellar_rhod = np.zeros(nb_stars, dtype="f4")
                stellar_xion = np.zeros(nb_stars, dtype="f4")
                stellar_temp = np.zeros(nb_stars, dtype="f4")
                stellar_Z = np.zeros(nb_stars, dtype="f4")
                stellar_age = np.zeros(nb_stars, dtype="f4")
                stellar_fluxes = np.zeros(nb_stars, dtype="f4")
                stellar_lintr = np.zeros(nb_stars, dtype="f8")
                stellar_masses = np.zeros(nb_stars, dtype="f4")
                stellar_radius = np.zeros(nb_stars, dtype="f4")
                stellar_mag = np.zeros((len(att_sets), nb_stars), dtype="f4")
                stellar_beta = np.zeros((len(att_sets), nb_stars), dtype="f4")
                stellar_fesc = np.zeros((len(att_sets), nb_stars), dtype="f8")

                if rank == 0:
                    print("Loaded files")
                # get_overstep_hydro_cubed(big_dust,subcube_nb,snap_box_path,'dust',outside,n_subcubes=n_subcubes,size=sub_side,overstep=overstep)
                # temp_units

                ind_star = 0
                if test:
                    halo_fescs = []

                # if test: sub_halo_tab = sub_halo_tab[:1]
                for ind, halo in enumerate(sub_halo_tab):
                    if test:
                        print("Halo #%i" % ind)

                    r_px = halo["rpx"]

                    slices = np.index_exp[
                        lower_bounds[ind, 2] : upper_bounds[ind, 2],
                        lower_bounds[ind, 1] : upper_bounds[ind, 1],
                        lower_bounds[ind, 0] : upper_bounds[ind, 0],
                    ]

                    # if np.any(sm_rho == 0):
                    #     raise Exception("densities should never be 0")

                    # sm_rho = np.max([sm_rho, np.full_like(sm_rho, small_f)], axis=0)
                    # print(sm_rho.min())

                    sm_rho = big_rho[slices]
                    sm_rhod = big_rhod[slices]
                    sm_metals = big_metals[slices]
                    sm_xion = big_xion[slices]
                    sm_xHI = 1 - sm_xion
                    # sm_temp = big_temp[slices]
                    sm_temp = (
                        convert_temp(big_temp[slices], sm_rho, sm_xion) * temp_fact
                    )

                    if np.prod(sm_rho.shape) < 1:
                        continue

                    sm_taus = np.zeros(((len(att_sets),) + np.shape(sm_xHI)))

                    for iset, att_set in enumerate(att_sets):
                        sm_taus[iset] = ((sm_xHI * sm_rho) * (rho_fact * tau_fact)) + (
                            (sm_rhod * att_set.Kappa912) * (rhod_fact * px_to_m * 100.0)
                        )
                    sm_taus_dust = sm_taus[:] - sm_taus[0]

                    # print(sm_taus[0].max(), sm_taus[0].mean(), np.median(sm_taus[0]))

                    # print(sm_taus.shape)

                    # arg=np.unravel_index([np.argmax(sm_taus_dust[1,:])], sm_rho.shape)
                    # m=(sub_halo_tab["mass"][ind] * Mp)
                    # if m >1e10:print("%e"%m,sm_taus[0][arg],sm_taus_dust[1][arg],sm_rho[arg]*sm_xHI[arg]*rho_fact*Pmass*1e3, sm_rhod[arg]*rho_fact*Pmass*1e3)

                    # Xs, Ys, Zs = sph_2_cart(rs, Phis, Thes)

                    # X_circ,Y_circ,Z_circ=sph_2_cart(r_px,phis,thes)

                    # If there aren't any stars : no need to calculate emissivities or star formattion stuff
                    if sub_halo_star_nb[ind] > 0:
                        # Get stars for halo  from list of associated stars
                        cur_star_ids = halo_star_ids[
                            sub_halo_tot_star_nb[ind]
                            - sub_halo_star_nb[ind] : sub_halo_tot_star_nb[ind]
                        ]

                        nb_cur_stars = len(cur_star_ids)

                        if test:
                            print(f"Halo {ind:d} has {nb_cur_stars:d} stars")

                        rs = np.arange(
                            0, 2 * r_px, rad_res
                        )  # so we do edge cases properly

                        Rs, Phis = np.meshgrid(rs, phis)  # for healpix
                        Rs, Thes = np.meshgrid(rs, thes)  # for healpix
                        # Rs,Phis,Thes=np.meshgrid(rads,phis,thes) #for a meshgrid generated grid
                        Xs, Ys, Zs = sph_2_cart(rs, Phis, Thes)
                        # print(cur_star_ids)

                        # print(sub_halo_tot_star_nb[ind] - sub_halo_star_nb[ind], sub_halo_tot_star_nb[ind])

                        cur_stars = read_specific_stars(
                            os.path.join(star_path, output_str), cur_star_ids
                        )

                        halo_fluxes = (
                            cur_stars["mass"]
                            / (1 - eta_sn)
                            * 10
                            ** (
                                -get_star_mags_metals(
                                    cur_stars["age"],
                                    cur_stars["Z/0.02"] * 0.02,
                                    mags_fct,
                                )
                                / 2.5
                            )
                        )

                        cur_star_luminosity = (
                            10
                            ** (
                                get_star_xis_metals(
                                    cur_stars["age"],
                                    cur_stars["Z/0.02"] * 0.02,
                                    xis_fct,
                                )
                            )
                            * cur_stars["mass"]
                            / (1 - eta_sn)
                        )  # ph/s

                        low_conts = get_star_mags_metals(
                            cur_stars["age"], cur_stars["Z/0.02"] * 0.02, low_mags_fct
                        )
                        high_conts = get_star_mags_metals(
                            cur_stars["age"], cur_stars["Z/0.02"] * 0.02, high_mags_fct
                        )

                        emissivity_box = np.zeros_like(sm_rho, dtype=np.float64)

                        fracd_pos = np.transpose(
                            [cur_stars[key] for key in ["x", "y", "z"]]
                        )

                        fracd_pos *= ldx

                        # print(pos[ind],  ldx)
                        # print(fracd_pos)

                        fracd_pos += pos_vects[
                            np.argmin(
                                get_mult_27(pos[ind], fracd_pos, pos_vects), axis=1
                            )
                        ]

                        # print(pos[ind])
                        # print(fracd_pos)

                        # basically we get the indices of stars for a posittion histogram
                        sm_xgrid = np.arange(
                            lower_bounds_fesc[ind, 0], upper_bounds_fesc[ind, 0], 1
                        )
                        star_sm_posx = (
                            np.digitize(
                                fracd_pos[:, 0], sm_xgrid - edge_overstep + edge[0]
                            )
                            - 1
                        )

                        sm_ygrid = np.arange(
                            lower_bounds_fesc[ind, 1], upper_bounds_fesc[ind, 1], 1
                        )
                        star_sm_posy = (
                            np.digitize(
                                fracd_pos[:, 1], sm_ygrid - edge_overstep + edge[1]
                            )
                            - 1
                        )

                        sm_zgrid = np.arange(
                            lower_bounds_fesc[ind, 2], upper_bounds_fesc[ind, 2], 1
                        )
                        star_sm_posz = (
                            np.digitize(
                                fracd_pos[:, 2], sm_zgrid - edge_overstep + edge[2]
                            )
                            - 1
                        )

                        # Using indices we can sum up all the emissities in every cell of our halo
                        for istar, star in enumerate(cur_stars):
                            # print(star_sm_posz[istar],star_sm_posy[istar],star_sm_posx[istar], istar, cur_star_luminosity[istar])
                            emissivity_box[
                                star_sm_posz[istar],
                                star_sm_posy[istar],
                                star_sm_posx[istar],
                            ] += cur_star_luminosity[istar]

                        # print(np.argmax(emissivity_box), np.max(emissivity_box), emissivity_box.shape)
                        # print(np.unravel_index(np.argmax(emissivity_box), (emissivity_box.shape)))

                        smldx = np.shape(emissivity_box)[0]
                        xind, yind, zind = np.mgrid[0:smldx, 0:smldx, 0:smldx]

                        in_bounds = np.linalg.norm(
                            [
                                xind - 0.5 * smldx + 0.5,
                                yind - 0.5 * smldx + 0.5,
                                zind - 0.5 * smldx + 0.5,
                            ],
                            axis=0,
                        ) < (r_px)
                        # print(in_bounds)
                        cond = (
                            emissivity_box != 0
                        ) * in_bounds  # need to check that cell centre is in r200 even if stars won't be outside of r200

                        normed_emissivity_box = emissivity_box / np.sum(emissivity_box)

                        # sm_emissivity_box = normed_emissivity_box

                        cells_w_stars = normed_emissivity_box[cond]
                        xind, yind, zind = xind[cond], yind[cond], zind[cond]

                        dust_taus = np.zeros(
                            ((len(att_sets),) + emissivity_box.shape),
                            dtype=np.float32,
                        )

                        tau_box = np.zeros(
                            ((len(att_sets),) + emissivity_box.shape),
                            dtype=np.float64,
                        )

                        # print(cells_w_stars, len(cur_stars['mass']))

                        # if test and halo["mass"]*Mp<1E10:continue

                        for icell, (
                            cell_w_stars,
                            x_cell,
                            y_cell,
                            z_cell,
                        ) in enumerate(zip(cells_w_stars, xind, yind, zind)):
                            sm_ctr = np.asarray([z_cell, y_cell, x_cell]) + 0.5

                            # print(sm_taus[iset])
                            # print(np.shape(sm_taus), np.shape(sm_rho), np.shape(sm_rho))

                            for iset in range(len(att_sets)):
                                tau_box[
                                    iset, x_cell, y_cell, z_cell
                                ] += sum_over_rays_bias(
                                    sm_taus[iset],
                                    sm_ctr,
                                    r_px,
                                    rad_res,
                                    Xs,
                                    Ys,
                                    Zs,
                                    debug=False,
                                )  # * cell_w_stars

                            dust_taus[
                                :, x_cell, y_cell, z_cell
                            ] = shoot_star_path_cheap_multid(
                                sm_ctr, sm_taus_dust[:], 2 * r_px
                            )

                        # now we can use our indices again to get the proper tau/trans for every star : SO MUCH MUCH MUCH MUCH FASTER !
                        star_taus = dust_taus[
                            :, star_sm_posz, star_sm_posy, star_sm_posx
                        ]

                        star_taus_fesc = (
                            tau_box[:, star_sm_posz, star_sm_posy, star_sm_posx]
                            * normed_emissivity_box[
                                star_sm_posz, star_sm_posy, star_sm_posx
                            ]
                        )

                        stellar_radius[ind_star : ind_star + nb_cur_stars] = np.sqrt(
                            (star_sm_posz - sample_r_fesc[ind]) ** 2
                            + (star_sm_posy - sample_r_fesc[ind]) ** 2
                            + (star_sm_posx - sample_r_fesc[ind]) ** 2
                        )

                        if test:
                            print(np.max(sm_taus[0]), np.max(tau_box[0]))
                            print(np.argmax(sm_taus[0]), np.argmin(tau_box[0]))
                            print(np.sum(tau_box[0] * normed_emissivity_box))

                            vmin = 1e-4
                            vmax = 1.0

                            fig, ax = plt.subplots(3, 5, figsize=(25, 10))

                            ax[0, 0].imshow(sm_taus[0, :, :, int(smldx / 2)])
                            # plt.colorbar()
                            img = ax[0, 1].imshow(
                                (tau_box[0, :, :, int(smldx / 2)]),
                                norm=LogNorm(vmin=vmin, vmax=vmax),
                            )
                            plt.colorbar(img)
                            img = ax[0, 2].imshow(
                                (sm_rho[:, :, int(smldx / 2)] * unit_d),
                                norm=LogNorm(vmin=1e-28, vmax=1e-24),
                            )
                            plt.colorbar(img)
                            img = ax[0, 3].imshow(
                                np.exp(-sm_taus[0, :, :, int(smldx / 2)]),
                                norm=LogNorm(vmin=vmin, vmax=vmax),
                            )
                            plt.colorbar(img)
                            ax[0, 4].imshow(
                                (normed_emissivity_box[:, :, int(smldx / 2)])
                            )

                            ax[1, 0].imshow(sm_taus[0, :, int(smldx / 2), :])
                            # plt.colorbar()
                            img = ax[1, 1].imshow(
                                (tau_box[0, :, int(smldx / 2), :]),
                                norm=LogNorm(vmin=vmin, vmax=vmax),
                            )
                            plt.colorbar(img)
                            img = ax[1, 2].imshow(
                                (sm_rho[:, int(smldx / 2), :] * unit_d),
                                norm=LogNorm(vmin=1e-28, vmax=1e-24),
                            )
                            plt.colorbar(img)
                            img = ax[1, 3].imshow(
                                np.exp(-sm_taus[0, :, int(smldx / 2), :]),
                                norm=LogNorm(vmin=vmin, vmax=vmax),
                            )
                            plt.colorbar(img)
                            ax[1, 4].imshow(
                                (normed_emissivity_box[:, int(smldx / 2), :])
                            )

                            ax[2, 0].imshow(sm_taus[0, int(smldx / 2), :, :])
                            # plt.colorbar()
                            img = ax[2, 1].imshow(
                                (tau_box[0, int(smldx / 2), :, :]),
                                norm=LogNorm(vmin=vmin, vmax=vmax),
                            )
                            plt.colorbar(img)
                            img = ax[2, 2].imshow(
                                (sm_rho[int(smldx / 2), :, :] * unit_d),
                                norm=LogNorm(vmin=1e-28, vmax=1e-24),
                            )
                            plt.colorbar(img)
                            img = ax[2, 3].imshow(
                                np.exp(-sm_taus[0, int(smldx / 2), :, :]),
                                norm=LogNorm(vmin=vmin, vmax=vmax),
                            )
                            plt.colorbar(img)
                            ax[2, 4].imshow(
                                (normed_emissivity_box[int(smldx / 2), :, :])
                            )

                            ax[0, 0].set_title("tau")
                            ax[0, 1].set_title("trans")
                            ax[0, 2].set_title("rho")
                            ax[0, 3].set_title("trans (1 cell)")
                            ax[0, 4].set_title("fraction of emissivity")

                            # plot circle of size r_px on axis 3
                            circle = plt.Circle(
                                (smldx / 2, smldx / 2), r_px, color="r", fill=False
                            )

                            fig.savefig("test_tau_VS_ray")

                        print(
                            star_taus_fesc.min(),
                            star_taus_fesc.max(),
                            star_taus_fesc.mean(),
                        )

                        # print(np.argmax(tau_box[0]), np.argmax(normed_emissivity_box))

                        stellar_masses[ind_star : ind_star + nb_cur_stars] = cur_stars[
                            "mass"
                        ]  # missing /(1 - eta_sn) here...

                        # print(np.sum(stellar_masses[ind_star:ind_star + nb_cur_stars]), sub_halo_tab["mstar"][ind])

                        stellar_rho[ind_star : ind_star + nb_cur_stars] = (
                            sm_rho[star_sm_posz, star_sm_posy, star_sm_posx] * unit_d
                        )  # g.cm^-3
                        stellar_rhod[ind_star : ind_star + nb_cur_stars] = (
                            sm_rhod[star_sm_posz, star_sm_posy, star_sm_posx] * unit_d
                        )  # g.cm^-3
                        stellar_xion[ind_star : ind_star + nb_cur_stars] = sm_xion[
                            star_sm_posz, star_sm_posy, star_sm_posx
                        ]
                        stellar_taus[ind_star : ind_star + nb_cur_stars] = sm_taus[
                            0, star_sm_posz, star_sm_posy, star_sm_posx
                        ]
                        stellar_Z[ind_star : ind_star + nb_cur_stars] = (
                            cur_stars["Z/0.02"] * 0.02
                        )
                        stellar_age[ind_star : ind_star + nb_cur_stars] = cur_stars[
                            "age"
                        ]
                        stellar_fluxes[ind_star : ind_star + nb_cur_stars] = halo_fluxes
                        stellar_temp[ind_star : ind_star + nb_cur_stars] = sm_temp[
                            star_sm_posz, star_sm_posy, star_sm_posx
                        ]

                        for iset, att_set in enumerate(att_sets):
                            if att_set.Kappa912 > 0:
                                star_trans = np.exp(
                                    -star_taus[iset]
                                    * att_set.Kappa1600
                                    / att_set.Kappa912
                                )
                            else:
                                star_trans = np.ones_like(star_taus[0])

                            star_trans_fesc = star_taus_fesc[iset]

                            stellar_lintr[
                                ind_star : ind_star + nb_cur_stars
                            ] = cur_star_luminosity  # s^-1

                            stellar_mag[
                                iset, ind_star : ind_star + nb_cur_stars
                            ] = -2.5 * np.log10(halo_fluxes * star_trans)

                            stellar_beta[
                                iset, ind_star : ind_star + nb_cur_stars
                            ] = comp_betas_indv(
                                cur_stars["mass"] / (1 - eta_sn),
                                high_conts,
                                low_conts,
                                star_taus[iset],
                                att_set,
                            )

                            stellar_fesc[
                                iset, ind_star : ind_star + nb_cur_stars
                            ] = tau_box[0, star_sm_posz, star_sm_posy, star_sm_posx]

                        halo_mag = -2.5 * np.log10(np.sum(halo_fluxes))
                        halo_mag_ext = -2.5 * np.log10(
                            np.sum(
                                halo_fluxes
                                * np.exp(
                                    -star_taus[iset]
                                    * att_set.Kappa1600
                                    / att_set.Kappa912
                                )
                            )
                        )

                        wv_convert_high = att_sets[4].Kappa2500 / att_sets[4].Kappa912
                        wv_convert_low = att_sets[4].Kappa1500 / att_sets[4].Kappa912
                        delta_lambda = np.log10(2500.0 / 1500.0)

                        halo_beta = (
                            np.log10(
                                np.sum(high_conts * cur_stars["mass"] / (1 - eta_sn))
                                / np.sum(low_conts * cur_stars["mass"] / (1 - eta_sn))
                            )
                            / delta_lambda
                        )
                        halo_beta_ext = (
                            np.log10(
                                np.sum(
                                    high_conts
                                    * cur_stars["mass"]
                                    / (1 - eta_sn)
                                    * np.exp(-star_taus[4] * wv_convert_high)
                                )
                                / np.sum(
                                    low_conts
                                    * cur_stars["mass"]
                                    / (1 - eta_sn)
                                    * np.exp(-star_taus[4] * wv_convert_low)
                                )
                            )
                            / delta_lambda
                        )

                        print("mags & betas")
                        print(halo_mag, halo_beta)
                        print(halo_mag_ext, halo_beta_ext)

                        if test:
                            halo_fescs.append(
                                np.sum(
                                    stellar_lintr[ind_star : ind_star + nb_cur_stars]
                                    * stellar_fesc[
                                        0, ind_star : ind_star + nb_cur_stars
                                    ]
                                )
                                / np.sum(
                                    stellar_lintr[ind_star : ind_star + nb_cur_stars]
                                )
                            )

                            print(
                                stellar_fesc[
                                    iset, ind_star : ind_star + nb_cur_stars
                                ].min(),
                                stellar_fesc[
                                    iset, ind_star : ind_star + nb_cur_stars
                                ].max(),
                                stellar_fesc[
                                    iset, ind_star : ind_star + nb_cur_stars
                                ].mean(),
                            )
                            print(f"fesc is : {halo_fescs[-1]:.4e}")

                        ind_star += nb_cur_stars

                    # if test and halo["mstar"] > 0:
                    if test and halo["mass"] * Mp > 1.0e11:
                        pass

                # print(halo_2e4k_gmass, halo_2e4k_gmass.max(), np.mean(halo_2e4k_gmass))

                if not test:
                    pos = np.transpose([sub_halo_tab[key] for key in ["x", "y", "z"]])

                    print("Writing %s" % out_file)

                    write_fields(
                        out_file,
                        stellar_rho,
                        stellar_rhod,
                        stellar_xion,
                        stellar_temp,
                        stellar_age,
                        stellar_Z,
                        stellar_mag,
                        stellar_beta,
                        stellar_fesc,
                        stellar_masses,
                        stellar_lintr,
                        stellar_radius,
                        att_sets,
                    )

                else:
                    print("Test run, out_file: %s" % out_file)

                    print(stellar_fesc.min(), stellar_fesc.max(), stellar_fesc.mean())

                    # test fesc by plotting the fesc as a function of the stellar_rho

                    stellar_fesc[0][stellar_fesc[0] < 1e-10] = 1e-10

                    fig, axs = plt.subplots(
                        1, 2, figsize=(10, 10), sharey=True, width_ratios=[0.85, 0.15]
                    )
                    print(len(stellar_rho), len(stellar_fesc[0]))
                    # axs[0].scatter(stellar_rho, stellar_fesc[0], c=stellar_age, s=5, cmap='viridis')
                    # axs[0].scatter(stellar_rho, stellar_fesc[0], c=np.log10(np.exp(-stellar_taus)), s=5, cmap='viridis')
                    axs[0].scatter(
                        stellar_rho,
                        stellar_fesc[0],
                        c=np.log10(stellar_lintr),
                        s=5,
                        cmap="viridis",
                    )

                    # plot mean in bins
                    rho_bins = np.logspace(-28, -24, 8)
                    mean, bins, counts = binned_statistic(
                        stellar_rho,
                        stellar_fesc[0] * stellar_lintr,
                        statistic="sum",
                        bins=rho_bins,
                    )
                    sum, bins, counts = binned_statistic(
                        stellar_rho, stellar_lintr, statistic="sum", bins=rho_bins
                    )
                    mean = mean / sum

                    axs[0].plot(bins[:-1], mean, "r--", label="mean")

                    axs[1].hist(
                        stellar_fesc[0],
                        bins=np.logspace(-10, 1, 60),
                        orientation="horizontal",
                    )
                    axs[1].scatter(1.0, halo_fescs[-1], marker="*", color="r", s=100)
                    axs[1].set_xscale("log")

                    axs[0].set_xlabel(r"$\rho$")
                    axs[0].set_ylabel(r"$f_{esc}$")
                    axs[0].set_xscale("log")
                    axs[0].set_yscale("log")
                    # axs[0].set_ylim(np.min(mean[np.isfinite(mean)]),1)

                    fig.savefig("test_fesc_stars.png")

                    # plot fesc as a function of taus
                    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
                    print(len(stellar_rho), len(stellar_fesc[0]))
                    axs[0].scatter(
                        stellar_taus,
                        stellar_fesc[0],
                        c=stellar_age,
                        s=5,
                        cmap="viridis",
                    )
                    axs[1].scatter(
                        stellar_rho,
                        np.exp(-stellar_taus),
                        c=stellar_age,
                        s=5,
                        cmap="viridis",
                    )

                    axs[0].set_xlabel(r"$\tau$")
                    axs[0].set_ylabel(r"$f_{esc}$")
                    axs[0].set_xscale("log")
                    axs[0].set_yscale("log")

                    axs[1].set_xlabel(r"$\rho$")
                    axs[1].set_ylabel(r"$\tau$")
                    axs[1].set_xscale("log")
                    axs[1].set_yscale("log")

                    fig.savefig("test_fesc_taus.png")

                    # corner plot WITH SEABORN between all stellar_ quantities
                    sns.set(style="ticks", color_codes=True)

                    df = pd.DataFrame(
                        {
                            "rho": stellar_rho.tolist(),
                            "xion": stellar_xion.tolist(),
                            "age": stellar_age.tolist(),
                            "Z": stellar_Z.tolist(),
                            "fesc": stellar_fesc[0].tolist(),
                            "mass": stellar_masses.tolist(),
                            "lintr": stellar_lintr.tolist(),
                            "radius": stellar_radius.tolist(),
                        }
                    )

                    g = sns.PairGrid(df, diag_sharey=False, corner=True)
                    g.map_lower(sns.scatterplot)
                    # g.map_upper(plt.scatter)
                    g.map_diag(sns.histplot, lw=3)
                    plt.savefig("test_fesc_corner.png")

                    return (
                        stellar_Z,
                        stellar_age,
                        sub_halo_tab,
                        stellar_fluxes,
                        stellar_mag,
                        stellar_beta,
                        stellar_fesc,
                        sub_halo_tab["mass"] * Mp,
                        stellar_lintr,
                    )


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
        "--mbin", metavar="mbin", type=float, help="halo mass bin centre", default=1
    )
    Arg_parser.add_argument(
        "--mbin_width",
        metavar="mbin_width",
        type=float,
        help="halo mass bin centre",
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
        "--test",
        action="store_true",
        help="When used, code runs on one subcube and doesn't write",
        default=False,
    )
    Arg_parser.add_argument(
        "--mp",
        action="store_true",
        help="Use Mei Palanque's segmented halo catalogue",
        default=False,
    )

    args = Arg_parser.parse_args()

    out_nb = args.nb
    rtwo_fact = args.rtwo_fact
    assoc_mthd = args.assoc_mthd
    ll = args.ll
    overwrite = args.overwrite

    compute_part_pties(
        out_nb,
        rtwo_fact=rtwo_fact,
        assoc_mthd=assoc_mthd,
        ll=ll,
        overwrite=overwrite,
        test=args.test,
        mbin=args.mbin,
        mbin_width=args.mbin_width,
        mp=args.mp,
    )
