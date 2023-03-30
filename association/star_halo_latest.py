"""
------
Read halo and star files and make associations
Saved as binary file that repeats for N halos and another that contains M stars
Associated stars and non associated stars are saved to separate binary files
Two files track each halo star number and 1st star file posittion
A final file saves halo IDs
------

This verstion includes halos that are close to each other and/or without stars
It also tracks stars that don't end up associated, that are saved to separate 'LONE' star files

"""


import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from read_phew import read_phew
from tionread_stars import read_rank_star_files
from scipy import spatial
import time
import string
import argparse
import os
from utils.functions_latest import *
from utils.utils import *

# from read_fof import o_fof
import h5py
from utils.output_paths import *
from files.wrap_boxes import *
from scipy.stats import binned_statistic_dd
import h5py
from mpi4py import MPI


def find_nearby_stars(star_tree, r_px, ctr):

    return (ctr, star_tree.query_ball_point(ctr, r_px))


def find_nearby_stars_stellar(
    star_coords, star_masses, star_tree, r_px, ctr, ldx, mvt=0, overstep=False
):

    loc_mvt = 0.6
    mvt = 0

    ##print(mvt,loc_mvt)
    while loc_mvt > 0.5 or mvt < 0.5 * r_px:

        big_ball = star_tree.query_ball_point(ctr, r_px)
        cur_star_coords = star_coords[big_ball, :]
        cur_star_masses = star_masses[big_ball]

        l = len(cur_star_masses)
        if l < 1:
            return (ctr, [])
        elif l == 1:
            stellar_barycentre = cur_star_coords[0]
        elif not overstep:  # if we don't account for cases where halo is on edge of box
            stellar_barycentre = np.average(
                cur_star_coords, weights=cur_star_masses, axis=0
            )
        else:
            # print(overstep,r_px)
            # stellar_barycentre=get_ctr_mult(cur_star_coords,pos_vects,cur_star_masses)
            stellar_barycentre = get_ctr_mult_cheap(
                ctr, cur_star_coords, ldx, cur_star_masses
            )
            # print(cur_star_coords)
            # print(stellar_barycentre)
            # print(ctr)

        loc_mvt = np.linalg.norm([stellar_barycentre - ctr])

        mvt += loc_mvt

        ctr = stellar_barycentre

    return (ctr, big_ball)


def find_max_stellar(
    star_coords, star_masses, star_tree, r_px, ctr, ldx, mvt=0, overstep=False
):

    loc_mvt = 0.6
    mvt = 0
    stellar_peak = np.copy(ctr)
    # print('*****')
    while loc_mvt > 0.5 and mvt < 0.5 * r_px:

        ctr = np.copy(stellar_peak)

        big_ball = star_tree.query_ball_point(ctr, r_px)
        cur_star_coords = star_coords[big_ball, :]
        cur_star_masses = star_masses[big_ball]

        n_found = len(cur_star_masses)
        if n_found < 1:
            return (ctr, [])

        l = np.ceil(r_px) + 2
        bin_size = 1.0
        coord_bins = np.arange(-l, l, bin_size)

        # print(coord_bins,coord_bins-0.5*l+ctr[0],coord_bins-0.5*l+ctr[1],coord_bins-0.5*l+ctr[2])
        map_coords = np.asarray(
            [coord_bins + ctr[0], coord_bins + ctr[1], coord_bins + ctr[2]]
        )
        stellar_density_map, edges, numbers = binned_statistic_dd(
            cur_star_coords, cur_star_masses, "sum", bins=map_coords
        )

        # stellar_max=np.argmax(stellar_density_map)
        # print(r_px)
        # print(stellar_max, stellar_density_map, stellar_density_map.max())
        # stellar_peak_args=np.unravel_index(stellar_max, stellar_density_map.shape)
        # print(stellar_peak_args)
        # stellar_peak =map_coords[stellar_peak_args[0], stellar_peak_args[1], stellar_peak_args[2]]

        # #assert abs(np.sum(stellar_density_map)-np.sum(cur_star_masses))<1e2, print(np.sum(stellar_density_map),np.sum(cur_star_masses))

        # print(stellar_peak)

        # #print(np.shape(stellar_density_map), np.where(stellar_density_map==stellar_max))

        stellar_max = np.max(stellar_density_map)
        max_bools = stellar_density_map == stellar_max
        whs = np.where(max_bools)[0]
        if len(whs) > 1:
            stellar_peaks = [wh * bin_size - 0.5 * l + ctr for wh in whs]
            # mvts=[np.linalg.norm([stellar_peak-ctr]) for stellar_peak in stellar_peaks]
            # whs=whs[np.argmin(mvts)]
            stellar_peak = np.mean(stellar_peaks, axis=0)
            # if r_px>1:
            #         print('issue')
            #         print(stellar_peak,ctr)

        else:
            stellar_peak = whs * bin_size - 0.5 * l + ctr

        loc_mvt = np.linalg.norm([stellar_peak - ctr])

        mvt += loc_mvt

        # print(mvt, loc_mvt, r_px)
        # print(ctr)

    return (ctr, big_ball)


def find_nearby_stars_wrapper(
    star_coords, star_masses, star_tree, r_px, ctr, ldx, assoc_mthd, overstep
):

    if assoc_mthd == "" or assoc_mthd=="fof_ctr":
        ctr, bb = find_nearby_stars(star_tree, r_px, ctr)

    elif assoc_mthd == "star_barycentre":
        # pos_vects=gen_pos_vects(ldx)
        ctr, bb = find_nearby_stars_stellar(
            np.asarray(star_coords),
            np.asarray(star_masses),
            star_tree,
            r_px,
            ctr,
            ldx,
            0.0,
            overstep,
        )
    elif assoc_mthd == "stellar_peak":
        # pos_vects=gen_pos_vects(ldx)
        ctr, bb = find_max_stellar(
            np.asarray(star_coords),
            np.asarray(star_masses),
            star_tree,
            r_px,
            ctr,
            ldx,
            0.0,
            overstep,
        )

    return (ctr, bb)


def assoc_stars_to_haloes(
    out_nb,
    ldx,
    path,
    sim_name,
    use_fof=True,
    rtwo_fact=1,
    fof_path=None,
    npart_thresh=50,
    assoc_mthd="",
    overwrite=False,
):

    check_assoc_keys(assoc_mthd)

    # phew_path='/data2/jlewis/dusts/output_00'+out_nb
    # star_path='/data2/jlewis/dusts/'
    # info_path='/data2/jlewis/dusts/output_00'+out_nb

    output_str = "output_%06i" % out_nb

    info_path = os.path.join(path, output_str, "group_000001")
    star_path = os.path.join(path, "reduced", "stars", output_str)
    # phew_path=os.path.join(path,output_str)

    fof_suffix = get_fof_suffix(fof_path)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix, rtwo_suffix)

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

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    if rank == 0:
        # if no out folder, make it
        if not os.path.exists((assoc_out)):
            os.makedirs(assoc_out)

        # if no out folder, make it
        if not os.path.exists((analy_out)):
            os.makedirs(analy_out)

    out_file = os.path.join(assoc_out, ("assoc_halos_%s" % out_nb) + suffix)
    if os.path.exists(out_file) and not overwrite:
        if rank == 0:
            print("File %s exists and I'm not in overwrite mode ... Exiting"%out_file)
        return ()

    # Mp=3.*(H0*1e3/(pc*1e6))**2.*(om_m-om_b)*(Lco*pc*1e6)**3./Np_tot/8./np.pi/G/Msol #En Msol 4.04*1e5
    Mp = get_Mp(om_m, om_b, H0, Lco, Np_tot)
    if rank == 0:
        print("Particle mass is :", Mp)

    if not "Mp" in os.listdir(out):
        out_Mp = open(os.path.join(out, "Mp"), "wb")
        out_Mp.write(np.float64(Mp))
        out_Mp.close()

    halos = []

    with h5py.File(os.path.join(fof_path, output_str, "haloes_masst.h5"), "r") as src:

        keys = src["Data"].keys()

        keys = [k for k in keys if k != "file_number"]

        for key in keys:
            halos.append(src["Data"][key][()])

    halos = np.asarray(halos).T

    # print(np.log10(np.max(halos[:,2])*Mp),get_r200(np.max(halos[:,2])))
    # print(np.log10(np.min(halos[:,2])*Mp),get_r200(np.min(halos[:,2])))

    new_ctrs = np.zeros((len(halos), 3))

    # trim <npart_thresh particle halos
    if npart_thresh > 50.0:
        halos = halos[halos[:, 1] >= npart_thresh, :]

    nhalos = len(halos)

    if rank == 0:
        print("Found %i haloes with npart>%i" % (nhalos, max(npart_thresh, 50)))

    halo_star_nb = np.zeros(nhalos)
    halo_stellar_mass = np.zeros(nhalos)
    halo_star_ids = []
    halo_halo_ids = []

    # Stars
    time_myr, stars, star_ids = read_rank_star_files(star_path, rank, Nproc)

    if rank == 0:
        print("read stars")

    families = stars["family"]

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
    stars = stars[families == 2, :]

    stellar_coords = stars[:, 1:4] * (ldx)
    stellar_coords[stellar_coords > ldx] -= ldx
    star_tree = spatial.cKDTree(stellar_coords, boxsize=ldx + 1e-6)

    # Local star IDs for creating list of 'lone stars'
    size = np.asarray(np.int32(len(halos)))

    stt = time.time()

    # Associattion and output

    r_pxs = get_r200(halos[:, 1], om_m) * rtwo_fact

    # new_halo_tab=np.c_[halos[:,1:],r_pxs]

    # print(np.max(halos[:,2]) * Mp)

    l_halos = len(halos)

    # nmin, nmax, nperProc = divide_task(len(halos), Nproc, rank)
    # halos_task = halos[nmin:nmax]
    # halo_task_nbs = np.arange(nmin, nmax)

    # for halo_nb,halo in zip(halo_task_nbs, halos_task):

    # print(r_px)

    if rank == 0:
        print("There are %i stars to match to %i haloes" % (len(stars), l_halos))
        print("Associattion starting")

    for halo_nb, halo in enumerate(halos[:]):

        r_px = r_pxs[halo_nb]

        ctr_vanilla = halo[2:5]  # ctr without rounding
        ctr = halo[2:5] * (ldx)

        # print(ctr)

        overstep = np.any(ctr + r_px > ldx) or np.any(ctr - r_px < 0)

        # Find nearby stars
        new_ctr, big_ball = find_nearby_stars_wrapper(
            stellar_coords,
            stars[:, 0],
            star_tree,
            r_px,
            ctr,
            ldx,
            assoc_mthd,
            overstep=overstep,
        )
        found_star_mass = stars[big_ball, 0]

        nb_stars = len(found_star_mass)

        # if np.any(new_ctr > ldx) :
        #         print(ctr, new_ctr)

        if nb_stars < 1:
            continue

        else:

            halo_star_ids.append(star_ids[big_ball])
            halo_halo_ids.append(halo_nb * np.ones(nb_stars))

            new_ctrs[halo_nb, :] = new_ctr[:]

            halo_star_nb[halo_nb] = nb_stars
            halo_stellar_mass[halo_nb] = np.sum(found_star_mass)

            # print("%e, %e, %f"%(halo[2] * Mp, halo_stellar_mass[halo_nb], r_px))

        if ((halo_nb) % 5000) == 0 and rank == 0:
            print(halo_nb / float(l_halos) * 100.0)

    # print('Done')

    comm.Barrier()

    halo_star_nb = sum_arrays_to_rank0(comm, halo_star_nb)
    halo_star_ids = merge_arrays_rank0(comm, np.concatenate(halo_star_ids))
    halo_halo_ids = merge_arrays_rank0(comm, np.concatenate(halo_halo_ids))
    halo_stellar_mass = sum_arrays_to_rank0(comm, halo_stellar_mass)
    norm = sum_arrays_to_rank0(comm, np.int8(np.all(new_ctrs != 0, axis=1)))
    new_ctrs = sum_arrays_to_rank0(comm, new_ctrs)  #

    comm.Barrier()

    if rank == 0:

        print("Associattion Finished")

        # we need to merge the work of the different processes by using our list of star ids and corresponding halo ids

        sort_halo_ids = np.argsort(halo_halo_ids)
        halo_star_ids = halo_star_ids[sort_halo_ids]

        non_null = norm > 0

        new_ctrs[non_null] = (
            new_ctrs[non_null] / norm[non_null, np.newaxis]
        )  # if halo stars are split accross processes, new_ctr can be >ldx
        # so we must weight it correctly to get the mean of the non-zero values from each process (the ones where there were stars in the halo and new_ctr is thus != 0)

        print(np.max(new_ctrs))

        with h5py.File((out_file), "w") as out_halos:

            out_halos.create_dataset(
                "ID", data=halos[:, 0], dtype=np.int64, compresstion="lzf"
            )
            out_halos.create_dataset(
                "mass", data=halos[:, 1], dtype=np.float32, compresstion="lzf"
            )
            out_halos.create_dataset(
                "coords", data=halos[:, 2:5], dtype=np.float32, compresstion="lzf"
            )
            out_halos.create_dataset(
                "rpx", data=r_pxs, dtype=np.float32, compresstion="lzf"
            )
            out_halos.create_dataset(
                "coords_new", data=new_ctrs[:, :], dtype=np.float32, compresstion="lzf"
            )
            out_halos.create_dataset(
                "stellar mass",
                data=halo_stellar_mass,
                dtype=np.float32,
                compresstion="lzf",
            )
            out_halos.create_dataset(
                "stellar count", data=halo_star_nb, dtype=np.int32, compresstion="lzf"
            )
            out_halos.create_dataset(
                "halo star ID", data=halo_star_ids, dtype=np.int64, compresstion="lzf"
            )

            print("Done")


"""
Main body
"""


if __name__ == "__main__":

    Arg_parser = argparse.ArgumentParser("Associate stars and halos in full simulattion")

    Arg_parser.add_argument(
        "nb",
        metavar="nsnap",
        type=int,
        help='snap number string, give as "XXX" so that 00001 is "001" and 00111 is "111"',
    )
    Arg_parser.add_argument("ldx", metavar="ldx", type=int, help="box size in cells")
    Arg_parser.add_argument("path", metavar="path", type=str, help="path to sim")
    Arg_parser.add_argument(
        "simname",
        metavar="simname",
        type=str,
        help="sim name (will be used to create dirs)",
    )
    Arg_parser.add_argument(
        "fofpath", metavar="fofpath", type=str, help="folder for fof masst hdf5 files"
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
        "--assoc_mthd",
        metavar="assoc_mthd",
        type=str,
        help="method for linking stars to fof",
        default="",
    )
    Arg_parser.add_argument(
        "--overwrite",
        acttion="store_true",
        help="overwrite existing files?",
        default=False,
    )

    args = Arg_parser.parse_args()

    out_nb = args.nb
    ldx = args.ldx
    path = args.path
    sim_name = args.simname
    fof_path = args.fofpath
    rtwo_fact = args.rtwo_fact
    npart_thresh = args.npart_thresh
    assoc_mthd = args.assoc_mthd

    assoc_stars_to_haloes(
        out_nb,
        ldx,
        path,
        sim_name,
        fof_path=fof_path,
        npart_thresh=npart_thresh,
        rtwo_fact=rtwo_fact,
        assoc_mthd=assoc_mthd,
        overwrite=args.overwrite,
    )
