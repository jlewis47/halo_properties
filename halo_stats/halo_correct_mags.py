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


def write_fields(
    Mp,
    att_sets,
    sub_halo_tab,
    out_file,
    sub_idxs,
    pos,
    halo_ray_Tr,
    halo_mags,
    halo_betas,
    halo_SFRs,
    halo_Lintrs,
    halo_stellar_mass,
    halo_youngest,
    halo_oldest,
    halo_stAgeWmass,
    halo_stZ_wStMass,
    halo_gmass,
    halo_2e4k_gmass,
    halo_max_rhog,
    halo_xhi_wV,
    halo_xhi_wMg,
    halo_xhi_wStMass,
    halo_gtemp_max,
    halo_gtemp_wMg,
    halo_gtemp_wStMass,
    halo_Zmass,
    halo_Zmass_wStMass,
    halo_Md,
    halo_Md_wStMass,
):
    with h5py.File(out_file, "w") as out_halos:
        out_halos.create_dataset("ID", data=sub_idxs, dtype=np.int64, compression="lzf")

        dset = out_halos.create_dataset(
            "mass",
            data=sub_halo_tab["mass"] * Mp,
            dtype=np.float32,
            compression="lzf",
        )
        dset.attrs['unit']='solar masses'
        dset.attrs['descripttion']='halo fof mass'

        dset = out_halos.create_dataset("pos", data=pos, dtype=np.float64, compression="lzf")
        dset.attrs['unit']='cells'

        for iset, att_set in enumerate(att_sets):
            trans_name = "Tr_%s" % att_set.name
            dset = out_halos.create_dataset(
                trans_name,
                data=halo_ray_Tr[iset, :],
                dtype=np.float32,
                compression="lzf",
            )
            dset.attrs['unit']=''
            dset.attrs['descripttion']='halo posittion'

            mag_name = "mag_%s" % att_set.name
            dset = out_halos.create_dataset(
                mag_name,
                data=halo_mags[iset, :],
                dtype=np.float32,
                compression="lzf",
            )

            dset.attrs['descripttion']='AB magnitude at 1600 Angstrom'
            dset.attrs['unit']=''

            beta_name = "betas_%s" % att_set.name
            dset = out_halos.create_dataset(
                beta_name,
                data=halo_betas[iset, :],
                dtype=np.float32,
                compression="lzf",
            )

            dset.attrs['unit']=''
            dset.attrs['descripttion']='UV slope'

        for ibin, name in enumerate(sfr_names):
            dset = out_halos.create_dataset(
                name,
                data=halo_SFRs[:, ibin],
                dtype=np.float32,
                compression="lzf",
            )

            dset.attrs['unit']='solar masses per Myr' 
            dset.attrs['descripttion']='time averaged sfr'           

            # print(halo_Lintrs[:,0])
        for ibin, name in enumerate(Lintr_names):
            dset = out_halos.create_dataset(
                name,
                data=halo_Lintrs[:, ibin],
                dtype=np.float64,
                compression="lzf",
            )

            dset.attrs['unit']='photons per second'
            dset.attrs['descripttion']='LyC luminosity'

        dset = out_halos.create_dataset(
            "Mst",
            data=halo_stellar_mass,
            dtype=np.float32,
            compression="lzf",
        )
        dset.attrs['unit']='solar masses'
        dset.attrs['descripttion']='stellar mass in rtwo_fact * r200'

        dset = out_halos.create_dataset(
            "oldest", data=halo_oldest, dtype=np.float32, compression="lzf"
        )
        dset.attrs['unit']='Myr'
        dset.attrs['descripttion']='age of oldest halo stellar particle'

        dset = out_halos.create_dataset(
            "youngest",
            data=halo_youngest,
            dtype=np.float32,
            compression="lzf",
        )
        dset.attrs['unit']='Myr'
        dset.attrs['descripttion']='age of youngest halo stellar particle'

        dset = out_halos.create_dataset(
            "stAge_wMst",
            data=halo_stAgeWmass,
            dtype=np.float32,
            compression="lzf",
        )
        dset.attrs['unit']='Myr'
        dset.attrs['descripttion']='stellar mass weighted stellar particle age'

        dset = out_halos.create_dataset(
            "stZ_wMst",
            data=halo_stZ_wStMass,
            dtype=np.float32,
            compression="lzf",
        )
        dset.attrs['unit']='absolute (not solar) metallicity'
        dset.attrs['descripttion']='stellar mass weighted stellar particle metallicity'

        dset = out_halos.create_dataset(
            "gmass", data=halo_gmass, dtype=np.float32, compression="lzf"
        )
        dset.attrs['unit']='solar masses'
        dset.attrs['descripttion']='halo gas mass'

        dset = out_halos.create_dataset(
            "gmass(2e4K)",
            data=halo_2e4k_gmass,
            dtype=np.float32,
            compression="lzf",
        )
        dset.attrs['unit']='solar masses'
        dset.attrs['descripttion']='halo gas mass of T<2e4K gas cells'

        dset = out_halos.create_dataset(
            "max(rho)",
            data=halo_max_rhog,
            dtype=np.float32,
            compression="lzf",
        )
        dset.attrs['unit']='H.cm^-3'
        dset.attrs['descripttion']='maximum halo density'

        dset = out_halos.create_dataset(
            "xhi_wMg",
            data=halo_xhi_wMg,
            dtype=np.float32,
            compression="lzf",
        )
        dset.attrs['unit']=''
        dset.attrs['descripttion']='gas mass weighted xhi'
        

        dset = out_halos.create_dataset(
            "xhi_wV", data=halo_xhi_wV, dtype=np.float32, compression="lzf"
        )
        dset.attrs['unit']=''
        dset.attrs['descripttion']='volume weighted xhi'

        dset = out_halos.create_dataset(
            "xhi_wMst",
            data=halo_xhi_wStMass,
            dtype=np.float32,
            compression="lzf",
        )
        dset.attrs['unit']=''
        dset.attrs['descripttion']='stellar mass weighted xhi'        
        

        dset = out_halos.create_dataset(
            "gtemp_max", data=halo_gtemp_max, dtype=np.float32, compression="lzf"
        )
        dset.attrs['unit']='K'
        dset.attrs['descripttion']='max gas temperature'

        dset = out_halos.create_dataset(
            "gtemp_wMg", data=halo_gtemp_wMg, dtype=np.float32, compression="lzf"
        )
        dset.attrs['unit']='K'
        dset.attrs['descripttion']='gas mass weighted average gas temperature'

        dset = out_halos.create_dataset(
            "gtemp_wMst",
            data=halo_gtemp_wStMass,
            dtype=np.float32,
            compression="lzf",
        )
        dset.attrs['unit']='K'
        dset.attrs['descripttion']='stellar mass weighted average gas temperature'

        dset = out_halos.create_dataset(
            "Zmass", data=halo_Zmass, dtype=np.float32, compression="lzf"
        )
        dset.attrs['unit']='solar masses'
        dset.attrs['descripttion']='halo metal mass'

        dset = out_halos.create_dataset(
            "Zmass_wMst",
            data=halo_Zmass_wStMass,
            dtype=np.float32,
            compression="lzf",
        )
        dset.attrs['unit']='solar masses'
        dset.attrs['descripttion']='stellar mass weighted average gas metallicity'

        dset = out_halos.create_dataset(
            "Md", data=halo_Md, dtype=np.float32, compression="lzf"
        )
        dset.attrs['unit']='solar masses'
        dset.attrs['descripttion']='halo dust mass'

        dset = out_halos.create_dataset(
            "Md_wMst",
            data=halo_Md_wStMass,
            dtype=np.float32,
            compression="lzf",
        )
        dset.attrs['unit']='solar masses'
        dset.attrs['descripttion']='stellar mass weighted average dust mass'


def compute_fesc(
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
        print("I found %i sets of dust absorpttion coefficients" % (len(att_sets) - 1))

    assert (
        att_sets[0].Kappa912 == 0.0
    ), "First attenuattion coeff set must be null -> no case dust case"

    big_side = int(sub_side * overstep)

    # # pre-allocate boxes for data cubes
    # # do this once here NOTE: since we don't reinitialise these arrays, the edges can look weird DON'T PANIC!!!! THIS IS THE PLAN
    # # the weird edges don't contain haloes and haven't been updated since the prevtions subcube
    # big_rho = np.memmap(
    #     os.path.join("tmp", "rho.dat"),
    #     shape=(big_side, big_side, big_side),
    #     dtype=np.float32,
    #     mode="w+",
    # )
    # big_rhod = np.memmap(
    #     os.path.join("tmp", "rhod.dat"),
    #     shape=(big_side, big_side, big_side),
    #     dtype=np.float32,
    #     mode="w+",
    # )
    # big_metals = np.memmap(
    #     os.path.join("tmp", "metals.dat"),
    #     shape=(big_side, big_side, big_side),
    #     dtype=np.float32,
    #     mode="w+",
    # )
    # big_xtion = np.memmap(
    #     os.path.join("tmp", "xtion.dat"),
    #     shape=(big_side, big_side, big_side),
    #     dtype=np.float32,
    #     mode="w+",
    # )
    # big_temp = np.memmap(
    #     os.path.join("tmp", "temp.dat"),
    #     shape=(big_side, big_side, big_side),
    #     dtype=np.float32,
    #     mode="w+",
    # )
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

                if test and subcube_nb >= Nproc:
                    continue

                if subcube_nb < fmin or subcube_nb >= fmax:
                    continue

                out_file = os.path.join(analy_out, "halo_stats_%i" % subcube_nb)

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

                limit_r_fesc = sub_halo_tab["rpx"] * fesc_rad + 1
                sample_r_fesc = do_half_round(limit_r_fesc)


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

                # dust and gas ray tracing
                halo_ray_Tr = np.zeros(
                    (len(att_sets), np.shape(sub_halo_tab)[0]), dtype=np.float64
                )
                halo_mags = np.zeros(
                    (len(att_sets), np.shape(sub_halo_tab)[0]), dtype=np.float32
                )
                halo_betas = np.zeros(
                    (len(att_sets), np.shape(sub_halo_tab)[0]), dtype=np.float32
                )

                # star pties
                halo_SFRs = np.zeros(
                    (np.shape(sub_halo_tab)[0], len(sfr_age_bins) - 1), dtype=np.float32
                )
                halo_Lintrs = np.zeros(
                    (np.shape(sub_halo_tab)[0], len(sfr_age_bins)), dtype=np.float64
                )

                halo_stellar_mass = np.zeros(
                    (np.shape(sub_halo_tab)[0]), dtype=np.float32
                )
                # halo_emissivity = np.zeros(
                    # (np.shape(sub_halo_tab)[0]), dtype=np.float64
                # )

                halo_youngest = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                halo_oldest = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                halo_stAgeWmass = np.zeros(
                    (np.shape(sub_halo_tab)[0]), dtype=np.float32
                )

                halo_stZ_wStMass = np.zeros(
                    (np.shape(sub_halo_tab)[0]), dtype=np.float32
                )

                # gas pties
                halo_gmass = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                halo_2e4k_gmass = np.zeros(
                    (np.shape(sub_halo_tab)[0]), dtype=np.float32
                )
                halo_max_rhog = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                # halo_rhog_avg = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                halo_xhi_wV = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                halo_xhi_wMg = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                halo_xhi_wStMass = np.zeros(
                    (np.shape(sub_halo_tab)[0]), dtype=np.float32
                )

                halo_gtemp_max = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                halo_gtemp_wMg = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                halo_gtemp_wStMass = np.zeros(
                    (np.shape(sub_halo_tab)[0]), dtype=np.float32
                )

                # metals
                halo_Zmass = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                halo_Zmass_wStMass = np.zeros(
                    (np.shape(sub_halo_tab)[0]), dtype=np.float32
                )
                # halo_avgZ = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                # halo_rhoZ_avg = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)

                # dust
                halo_Md = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                halo_Md_wStMass = np.zeros(
                    (np.shape(sub_halo_tab)[0]), dtype=np.float32
                )
                # halo_avgMd = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)
                halo_rhod_avg = np.zeros((np.shape(sub_halo_tab)[0]), dtype=np.float32)

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
                )
                # done = True
                # except IOError:
                # sleep(1)

                if rank == 0:
                    print("Loaded files")
                # get_overstep_hydro_cubed(big_dust,subcube_nb,snap_box_path,'dust',outside,n_subcubes=n_subcubes,size=sub_side,overstep=overstep)
                # temp_units

                for ind, halo in enumerate(sub_halo_tab):

                    # print('    Halo #%i'%ind)

                    r_px = halo["rpx"]


                    # print(f"r_px:{r_px:f} \n sample_r:{sample_r[ind]:f}")
                    # print(f"r_px:{r_px * fesc_rad:f} \n sample_r:{sample_r_fesc[ind]:f}")

                    slices = np.index_exp[
                        lower_bounds[ind, 2] : upper_bounds[ind, 2],
                        lower_bounds[ind, 1] : upper_bounds[ind, 1],
                        lower_bounds[ind, 0] : upper_bounds[ind, 0],
                    ]




                    # if np.any(sm_rho == 0):
                    #     raise Excepttion("densities should never be 0")

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

                    if fesc_rad != 1.0:

                        slices_fesc = np.index_exp[
                        lower_bounds_fesc[ind, 2] : upper_bounds_fesc[ind, 2],
                        lower_bounds_fesc[ind, 1] : upper_bounds_fesc[ind, 1],
                        lower_bounds_fesc[ind, 0] : upper_bounds_fesc[ind, 0],
                    ]
                        
                        sm_rho_fesc = big_rho[slices_fesc]
                        sm_rhod_fesc = big_rhod[slices_fesc]
                        sm_xHI_fesc = 1 - big_xion[slices_fesc]

                    else:

                        sm_rho_fesc = sm_rho
                        sm_rhod_fesc = sm_rhod
                        sm_xHI_fesc =sm_xHI



                    if np.prod(sm_rho.shape)<1:continue

                    # halo_gmass[ind] = (
                    #     np.sum(sm_rho)
                    #     * rho_fact
                    #     * (Pmass / Msol)
                    #     * (px_to_m * 1e2) ** 3
                    # )
                    # halo_rhog_avg[ind] = np.mean(sm_rho) * rho_fact
                    # halo_2e4k_gmass[ind] = (
                    #     np.sum(sm_rho[sm_temp > 2e4])
                    #     * rho_fact
                    #     * (Pmass / Msol)
                    #     * (px_to_m * 1e2) ** 3
                    # )
                    # halo_max_rhog[ind] = np.max(sm_rho) * rho_fact  # H/ccm

                    # halo_xhi_wMg[ind] = np.average(sm_xHI, weights=sm_rho)
                    # halo_xhi_wV[ind] = np.mean(sm_xHI)

                    # halo_gtemp[ind] = np.mean(sm_temp)

                    # halo_Md[ind] = (
                    #     np.sum(sm_rhod)
                    #     * rhod_fact
                    #     * (1e-3 / Msol)
                    #     * (px_to_m * 1e2) ** 3
                    # )
                    # halo_rhod_avg[ind] = np.mean(sm_rhod) * rhod_fact
                    # halo_Zmass[ind] = (
                    #     np.sum(sm_metals * sm_rho)
                    #     * rho_fact
                    #     * (Pmass / Msol)
                    #     * (px_to_m * 1e2) ** 3
                    # )
                    # halo_rhoZ_avg[ind] = np.mean(sm_rho * sm_metals) * rho_fact

                    #run faster by getting indices of relevent cells once (not once for evety mean/sum etc)
                    sample_coords = get_sample_coords(sm_rho.shape, dilate)

                    halo_gmass[ind] = (
                        sum_sph(sm_rho, r_px, sample_coords, dilate)
                        * rho_fact
                        * (Pmass / Msol)
                        * (px_to_m * 1e2) ** 3
                    )
                    # halo_rhog_avg[ind] = mean_sph(sm_rho, r_px, sample_coords, dilate) * rho_fact
                    
                    halo_2e4k_gmass[ind] = (
                        sum_sph(sm_rho, r_px, sample_coords, dilate, weights=sm_temp > 2e4)
                        * rho_fact
                        * (Pmass / Msol)
                        * (px_to_m * 1e2) ** 3
                    )
                    halo_max_rhog[ind] = np.max(sm_rho) * rho_fact  # H/ccm

                    halo_xhi_wMg[ind] = sum_sph(sm_xHI, r_px, sample_coords, dilate, weights=sm_rho/np.sum(sm_rho))
                    halo_xhi_wV[ind] = mean_sph(sm_xHI, r_px, sample_coords, dilate)

                    halo_gtemp_wMg[ind] = mean_sph(sm_temp, r_px, sample_coords, dilate, weights=sm_rho/np.sum(sm_rho))
                    halo_gtemp_max[ind] = np.max(sm_temp)

                    halo_Md[ind] = (
                        sum_sph(sm_rhod, r_px, sample_coords, dilate)
                        * rhod_fact
                        * (1e-3 / Msol)
                        * (px_to_m * 1e2) ** 3
                    )
                    # halo_rhod_avg[ind] = mean_sph(sm_rhod, r_px, sample_coords, dilate) * rhod_fact
                    halo_Zmass[ind] = (
                        sum_sph(sm_metals * sm_rho, r_px, sample_coords, dilate)
                        * rho_fact
                        * (Pmass / Msol)
                        * (px_to_m * 1e2) ** 3
                    )
                    # halo_rhoZ_avg[ind] = mean_sph(sm_rho * sm_metals, r_px, sample_coords, dilate) * rho_fact

                    sm_taus = np.zeros(((len(att_sets),) + np.shape(sm_xHI_fesc)))

                    for iset, att_set in enumerate(att_sets):
                        sm_taus[iset] = ((sm_xHI_fesc * sm_rho_fesc) * (rho_fact * tau_fact)) + (
                            (sm_rhod_fesc * att_set.Kappa912) * (rhod_fact * px_to_m * 100.0)
                        )  # includes partial dx
                    sm_taus_dust = sm_taus[:] - sm_taus[0]

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

                        rs = np.arange(0, 2 * r_px * fesc_rad, rad_res)  # so we do edge cases properly

                        Rs, Phis = np.meshgrid(rs, phis)  # for healpix
                        Rs, Thes = np.meshgrid(rs, thes)  # for healpix
                        # Rs,Phis,Thes=np.meshgrid(rads,phis,thes) #for a meshgrid generated grid
                        Xs, Ys, Zs = sph_2_cart(rs, Phis, Thes)
                        # print(cur_star_ids)

                        # print(sub_halo_tot_star_nb[ind] - sub_halo_star_nb[ind], sub_halo_tot_star_nb[ind])

                        cur_stars = read_specific_stars(
                            os.path.join(star_path, output_str), cur_star_ids
                        )

                        halo_stellar_mass[ind] = np.sum(cur_stars["mass"]) / (
                            1 - eta_sn
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

                        halo_stAgeWmass[ind] = np.average(
                            cur_stars["age"], weights=cur_stars["mass"]
                        )
                        halo_stZ_wStMass[ind] = np.average(
                            cur_stars["Z/0.02"] * 0.02, weights=cur_stars["mass"]
                        )

                        halo_oldest[ind] = np.max(cur_stars["age"])
                        halo_youngest[ind] = np.min(cur_stars["age"])

                        halo_SFRs[ind], halo_Lintrs[ind] = bin_star_info(
                            halo_SFRs[ind],
                            halo_Lintrs[ind],
                            cur_stars,
                            cur_star_luminosity,
                            sfr_age_bins,
                        )

                        emissivity_box = np.zeros_like(sm_rho_fesc, dtype=np.float64)

                    

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

                        in_bounds = (
                            np.linalg.norm(
                                [
                                    xind - 0.5 * smldx + 0.5,
                                    yind - 0.5 * smldx + 0.5,
                                    zind - 0.5 * smldx + 0.5,
                                ],
                                axis=0,
                            )
                            < (r_px * fesc_rad)
                        )
                        # print(in_bounds)
                        cond = (
                            emissivity_box != 0
                        ) * in_bounds  # need to check that cell centre is in r200 even if stars won't be outside of r200

                        normed_emissivity_box = emissivity_box / np.sum(
                            emissivity_box
                        )

                        if fesc_rad != 1.0:
                            delta = int(sample_r_fesc[ind] - sample_r[ind])

                            # print(sample_r_fesc[ind] , sample_r[ind], delta)

                            sm_emissivity_box = normed_emissivity_box[delta:-delta, delta:-delta, delta:-delta]
                            # print(sm_emissivity_box.shape, normed_emissivity_box.shape)
                            # print(sm_rho.shape, sm_rho_fesc.shape)

                        else:
                            sm_emissivity_box = normed_emissivity_box

                        cells_w_stars = normed_emissivity_box[cond]
                        xind, yind, zind = xind[cond], yind[cond], zind[cond]


                        # stellar mass weightred gas quantities
                        halo_gtemp_wStMass[ind] = np.sum(
                            sm_temp * sm_emissivity_box
                        )
                        halo_Md_wStMass[ind] = (
                            np.sum(sm_rhod * sm_emissivity_box)
                            * rho_fact
                            * (Pmass / Msol)
                            * (px_to_m * 1e2) ** 3
                        )
                        halo_Zmass_wStMass[ind] = (
                            np.sum(sm_metals * sm_rho * sm_emissivity_box)
                            * rho_fact
                            * (Pmass / Msol)
                            * (px_to_m * 1e2) ** 3
                        )
                        halo_xhi_wStMass[ind] = np.average(
                            sm_xHI, weights=sm_emissivity_box
                        )

                        dust_taus = np.zeros(
                            ((len(att_sets),) + emissivity_box.shape),
                            dtype=np.float32,
                        )

                        # print(cells_w_stars, len(cur_stars['mass']))

                        for icell, (
                            cell_w_stars,
                            x_cell,
                            y_cell,
                            z_cell,
                        ) in enumerate(zip(cells_w_stars, xind, yind, zind)):

                            sm_ctr = np.asarray([z_cell, y_cell, x_cell]) + 0.5

                            # print(sm_taus[iset])
                            print(np.shape(sm_taus), np.shape(sm_rho_fesc), np.shape(sm_rho))

                            halo_ray_Tr[:, ind] += (
                                sum_over_rays_bias_multid(
                                    sm_taus,
                                    sm_ctr,
                                    r_px * fesc_rad, #go further if fesc_rad > 1
                                    rad_res,
                                    Xs,
                                    Ys,
                                    Zs,
                                    debug=False,
                                )
                            ) * cell_w_stars


                            dust_taus[
                                :, x_cell, y_cell, z_cell
                            ] = shoot_star_path_cheap_multid(
                                sm_ctr, sm_taus_dust[:], 2 * r_px * fesc_rad
                            )

                        # now we can use our indices again to get the proper tau/trans for every star : SO MUCH MUCH MUCH MUCH FASTER !
                        star_taus = dust_taus[
                            :, star_sm_posz, star_sm_posy, star_sm_posx
                        ]

                        for iset, att_set in enumerate(att_sets):

                            if att_set.Kappa912 > 0.0:
                                star_trans = np.exp(
                                    -star_taus[iset]
                                    * att_set.Kappa1600
                                    / att_set.Kappa912
                                )
                            else:
                                star_trans = np.ones_like(star_taus[iset])

                            halo_mags[iset, ind] = -2.5 * np.log10(
                                np.nansum(halo_fluxes * star_trans)
                            )

                            halo_betas[iset, ind] = comp_betas(
                                cur_stars["mass"],
                                high_conts,
                                low_conts,
                                star_taus[iset],
                                att_set,
                            )

                    # if test and halo["mstar"] > 0:
                    if test and halo["mass"] * Mp > 1.0e11:
                        print("found a big halo")
                        print("%E" % (halo["mass"] * Mp))
                        print("gas mass=%E msun" % halo_gmass[ind])
                        print("dust mass=%E msun" % halo_Md[ind])
                        print("max(rhog)=%E Hpccm" % halo_max_rhog[ind])
                        print("extincttions", halo_mags[1:, ind] - halo_mags[0, ind])
                        print("mags", halo_mags[:, ind])
                        print("betas", halo_betas[:, ind])
                        print(np.max(dust_taus, axis=(1, 2, 3)))
                        # if halo_ray_Tr[0, ind] < 1.0:
                        #     fesc_debug_avg += halo_ray_Tr[0, ind]
                        #     debug_counts += 1.0
                        #     print("%e" % (fesc_debug_avg / debug_counts))

            # print(halo_2e4k_gmass, halo_2e4k_gmass.max(), np.mean(halo_2e4k_gmass))

                if not test:
                    pos = np.transpose([sub_halo_tab[key] for key in ["x", "y", "z"]])

                    print("Writing %s" % out_file)

        

                    write_fields(
                        Mp,
                        att_sets,
                        sub_halo_tab,
                        out_file,
                        sub_idxs,
                        pos,
                        halo_ray_Tr,
                        halo_mags,
                        halo_betas,
                        halo_SFRs,
                        halo_Lintrs,
                        halo_stellar_mass,
                        halo_youngest,
                        halo_oldest,
                        halo_stAgeWmass,
                        halo_stZ_wStMass,
                        halo_gmass,
                        halo_2e4k_gmass,
                        halo_max_rhog,
                        halo_xhi_wV,
                        halo_xhi_wMg,
                        halo_xhi_wStMass,
                        halo_gtemp_max,
                        halo_gtemp_wMg,
                        halo_gtemp_wStMass,
                        halo_Zmass,
                        halo_Zmass_wStMass,
                        # halo_avgZ,
                        halo_Md,
                        halo_Md_wStMass,
                        # halo_avgMd,
                    )

                else:

                    print('Test run, out_file: %s'%out_file)

                    return(halo_ray_Tr, halo_mags, sub_halo_tab, halo_gmass, sub_halo_tab["mass"] * Mp)

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
        "--fesc_rad",
        metavar="fesc_rad",
        type=float,
        help="1.0 -> use association radius as integration limit for fesc computation",
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

    compute_fesc(
        out_nb,
        rtwo_fact=rtwo_fact,
        fesc_rad=fesc_rad,
        assoc_mthd=assoc_mthd,
        ll=ll,
        overwrite=overwrite,
        test=args.test,
        dilate=dilate
    )
