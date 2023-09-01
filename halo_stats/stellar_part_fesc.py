"""

"""


import numpy as np
import matplotlib as mpl

mpl.use("Agg")
# from read_radgpu import o_rad_cube_big
# from tionread_stars import read_all_star_files
import string
import argparse
import os
import healpy as hp
import h5py
from scipy.stats import binned_statistic

# from halo_properties.association.read_assoc_latest import *

from halo_properties.files.wrap_boxes import read_cutout
from halo_properties.files.read_stars import read_star_file


from halo_properties.utils.functions_latest import get_infos
from halo_properties.utils.output_paths import *
from halo_properties.utils.units import *
from halo_properties.utils.utils import merge_arrays_rank0_Ndd_flatten_rebuild

# from halo_properties.utils.utils import get_fof_suffix, get_r200_suffix, get_suffix

# from halo_properties.src.bpass_fcts import *

from halo_properties.dust.att_coefs import *
from halo_properties.dust.dust_opacity import *
from halo_properties.dust.read_draine_tabs import *

from halo_properties.src.ray_fcts import *

from mpi4py import MPI


def compute_stellar_Tr(
    out_nb=-1, sub_nb=None, star_sub_l=2048, overd_fact=50, overwrite=False
):
    from halo_properties.params.params import star_path, pc, Pmass, ldx, nside, rad_res

    out_path = os.path.join(
        f"/gpfs/alpine/proj-shared/ast031/jlewis/{sim_name:s}" + "_analysis",
        "stellar_part_fesc",
        f"output_{out_nb:06d}",
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rstar_px_min = 1.0

    if rank == 0:
        os.makedirs(out_path, exist_ok=True)

    info_path = os.path.join(
        sim_path,
        f"output_{out_nb:06d}",
        "group_000001",
    )

    # Get scale factor and co
    """Get scale factor and co"""
    (
        t,
        aexp,
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

    H0_SI = H0 * 1000.0 / (pc * 1e6)  # in s^-1
    mean_b_density = (
        om_b * 3 * H0_SI**2 / (8 * np.pi * G) / Pmass * aexp**3
    )  # in H.m^-3

    rmax = 15
    rprof_bins = np.arange(0, rmax, 2)  # 2 cell resolution

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

    star_sub_p_side = ldx // star_sub_l

    if sub_nb == None:
        if rank == 0:
            print("no sub_nb given, computing all for sub boxes")
        subnbs = np.arange(0, (star_sub_p_side) ** 3)
    else:
        subnbs = np.array([sub_nb])

    rho_fact = get_unit_facts("rho", px_to_m, unit_d, unit_l, unit_t, aexp)
    rhod_fact = get_unit_facts("rhod", px_to_m, unit_d, unit_l, unit_t, aexp)
    tau_fact = get_unit_facts("tau", px_to_m, unit_d, unit_l, unit_t, aexp)

    for subnb in subnbs:
        out_file = os.path.join(out_path, f"stellar_part_fesc_{subnb:d}.hdf5")

        if os.path.isfile(out_file) and not overwrite:
            continue

        lower_corner = (
            np.asarray(
                np.unravel_index(
                    subnb, (star_sub_p_side, star_sub_p_side, star_sub_p_side)
                )
            )
        ) * star_sub_l

        higher_corner = lower_corner + star_sub_l

        xgrid = np.arange(lower_corner[0], higher_corner[0] + 1)
        ygrid = np.arange(lower_corner[1], higher_corner[1] + 1)
        zgrid = np.arange(lower_corner[2], higher_corner[2] + 1)

        # print(list(zip(lower_corner, higher_corner)))

        # print(xgrid.min(), xgrid.max())
        # print(ygrid.min(), ygrid.max())
        # print(zgrid.min(), zgrid.max())

        # load stars
        load_star_path = os.path.join(
            star_path, f"output_{out_nb:06d}", f"stars_{subnb:d}"
        )

        stars = np.array_split(read_star_file(load_star_path)[1], size)[rank]
        # could improve load balancing between tasks by making split on unique sets of coords instead of number of stars
        # currently some tasks have a 2x-0.33x workload... But this would drastically increase the memory requirements

        stx, sty, stz = (
            stars["x"],
            stars["y"],
            stars["z"],
        )

        # print(stars["x"].min() * ldx, stars["x"].max() * ldx)
        # print(stars["y"].min() * ldx, stars["y"].max() * ldx)
        # print(stars["z"].min() * ldx, stars["z"].max() * ldx)

        del stars

        # print(xgrid, ygrid, zgrid)

        xinds = np.digitize(stx * ldx, xgrid) - 1
        yinds = np.digitize(sty * ldx, ygrid) - 1
        zinds = np.digitize(stz * ldx, zgrid) - 1

        # print(xinds.min(), xinds.max())
        # print(yinds.min(), yinds.max())
        # print(zinds.min(), zinds.max())

        del stx, sty, stz

        # make unique indice sets
        stellar_cell_coords = np.unique(
            np.transpose(np.vstack((xinds, yinds, zinds))),
            axis=0,
        )

        # print(len(xinds), len(np.unique(xinds)), len(stellar_cell_coords))

        xinds, yinds, zinds = stellar_cell_coords.T

        nb_stellar_cells = len(xinds)

        print("rank %i found %i unique stellar cells" % (rank, nb_stellar_cells))

        stellar_cell_coords = np.transpose([xinds, yinds, zinds])

        fesc_grid = np.zeros((len(att_sets), nb_stellar_cells), dtype="f8")

        dust_tau_grid = np.zeros((len(att_sets), nb_stellar_cells), dtype="f8")

        cell_centers = (
            np.int32(np.array([xgrid[xinds], ygrid[yinds], zgrid[zinds]])) + 0.5
        )

        # Rs,Phis,Thes=np.meshgrid(rads,phis,thes) #for a meshgrid generated grid
        ctr_loc = np.array([rmax, rmax, rmax])  # + 0.5

        # iterate over cells with something in them

        sm_gridx, sm_gridy, sm_gridz = np.mgrid[: 2 * rmax, : 2 * rmax, : 2 * rmax]

        rs_prof = np.linalg.norm(
            np.asarray([np.ravel(sm_gridx), np.ravel(sm_gridy), np.ravel(sm_gridz)])
            - rmax,
            axis=0,
        )

        xarg, yarg, zarg = np.ravel(sm_gridx), np.ravel(sm_gridy), np.ravel(sm_gridz)

        for istar in range(nb_stellar_cells):
            cutouts = read_cutout(
                box_path + f"/output_{out_nb:06d}",
                ["rho", "xion", "dust"],
                np.asarray(cell_centers[:, istar]),
                size=rmax * 2,
            )

            sm_rho = cutouts[0] * rho_fact
            sm_xHI = cutouts[1]
            sm_rhod = cutouts[2] * rhod_fact

            sm_taus = np.zeros(((len(att_sets),) + np.shape(cutouts[0])))

            for iset, att_set in enumerate(att_sets):
                sm_taus[iset] = ((sm_xHI * sm_rho) * (rho_fact * tau_fact)) + (
                    (sm_rhod * att_set.Kappa912) * (rhod_fact * px_to_m * 100.0)
                )

            sm_taus_dust = sm_taus[:] - sm_taus[0]

            ctr = cell_centers[:, istar]

            # determine integration radius
            # furthest points to reach < 200 * mean_b_density

            # print(sm_rho.max(), sm_rho.max() / mean_b_density, mean_b_density)
            # print(sm_rho.min(), sm_rho.min() / mean_b_density)

            # wh_dens = np.where(sm_rho < overd_fact * mean_b_density)

            # rs_dens = np.linalg.norm(np.transpose(wh_dens) - ctr_loc, axis=1)

            rdens, bins, nbs = binned_statistic(
                rs_prof,
                sm_rho[xarg, yarg, zarg],
                "mean",
                bins=rprof_bins,
            )

            # print(rdens / mean_b_density)
            under_dens_rad = np.where(rdens < overd_fact * mean_b_density)

            if len(under_dens_rad[0]) > 0:
                # print(under_dens_rad)
                rstar_px = rprof_bins[np.min(under_dens_rad)]

                # print(
                #     rs_dens.min(),
                #     rs_dens.max(),
                #     rs_dens.mean(),
                #     np.median(rs_dens),
                #     rs_dens.std(),
                # )

                # radii are always the same... issue here!

                # rstar_px = np.median(rs_dens)

                # wh_closest_dens = np.argmax(np.abs(rs_dens))

                # print(
                #     sm_rho[
                #         wh_dens[0][wh_closest_dens],
                #         wh_dens[1][wh_closest_dens],
                #         wh_dens[2][wh_closest_dens],
                #     ]
                #     / mean_b_density,
                # )

                # rstar_px = rs_dens[wh_closest_dens]
                # rstar_px = min(rstar_px, rmax)
                rstar_px = np.nanmax([rstar_px, rstar_px_min])

            else:
                rstar_px = rstar_px_min

            # print(wh_closest_dens, rstar_px)

            rs = np.arange(0, 2 * rstar_px, rad_res)  # so we do edge cases properly
            Rs, Phis = np.meshgrid(rs, phis)  #  for healpix
            Rs, Thes = np.meshgrid(rs, thes)  # for healpix
            Xs, Ys, Zs = sph_2_cart(rs, Phis, Thes)

            # print(Xs, Ys, Zs)

            # some centering issue here.... not getting correct values when compared to the expected max value from starting cell
            # radii issues need some rethinking

            loc_fesc = sum_over_rays_bias_multid(
                sm_taus,
                ctr_loc,
                rstar_px,  # go further if fesc_rad > 1
                rad_res,
                Xs,
                Ys,
                Zs,
                debug=False,
            )

            # print(
            #     rmax,
            #     rstar_px,
            #     loc_fesc[0],
            #     # np.exp(-sm_taus[0, int(ctr_loc[0]), int(ctr_loc[1]), int(ctr_loc[2])]),
            # )

            loc_dust_tau = shoot_star_path_cheap_multid(
                ctr_loc, sm_taus_dust[:], 2 * rstar_px
            )

            fesc_grid[:, istar] = loc_fesc
            dust_tau_grid[:, istar] = loc_dust_tau

            if rank == 0 and istar % 1000 == 0:
                print(
                    istar / float(len(xinds)),
                    np.max(fesc_grid[0]),
                    np.min(fesc_grid[0, fesc_grid[0] > 0]),
                )

        # gather all the data
        # fesc_grid = np.concatenate(comm.gather(fesc_grid, root=0), axis=0)
        # dust_tau_grid = np.concatenate(comm.gather(dust_tau_grid, root=0), axis=0)
        # stellar_cell_coords = np.concatenate(
        #     comm.gather(stellar_cell_coords, root=0), axis=0
        # )
        comm.Barrier()
        # print(rank, stellar_cell_coords)
        # print(rank, fesc_grid[0, :])

        fesc_grid = merge_arrays_rank0_Ndd_flatten_rebuild(
            comm, fesc_grid.T, dtype="f8"
        )
        dust_tau_grid = merge_arrays_rank0_Ndd_flatten_rebuild(
            comm, dust_tau_grid.T, dtype="f8"
        )

        stellar_cell_coords = merge_arrays_rank0_Ndd_flatten_rebuild(
            comm, stellar_cell_coords, dtype="i8"
        )

        if rank == 0:
            print(stellar_cell_coords)
            print(fesc_grid[:, 0])

        nb_stellar_cells = np.sum(comm.gather(nb_stellar_cells, root=0))

        comm.Barrier()

        if rank == 0:
            stellar_cell_coords = stellar_cell_coords.T
            fesc_grid = fesc_grid.T
            dust_tau_grid = dust_tau_grid.T

            # print(nb_stellar_cells)

            # print(fesc_grid.shape)

            # print(fesc_grid[0])

            # print(stellar_cell_coords.shape, stellar_cell_coords)

            with h5py.File(out_file, "w") as dest:
                data_grp = dest.create_group("data")
                hdr_grp = dest.create_group("header")

                hdr_grp["nside"] = nside
                hdr_grp["rad_res"] = rad_res
                hdr_grp["rmax"] = rmax
                hdr_grp["ldx"] = ldx
                hdr_grp["sub_ldx"] = star_sub_l
                hdr_grp["sub_nb"] = subnb

                hdr_grp["out_nb"] = out_nb
                hdr_grp["aexp"] = aexp
                hdr_grp["sim_name"] = sim_name
                hdr_grp["sim_path"] = sim_path
                hdr_grp["star_path"] = star_path

                dest["help"] = [""]

                data_grp.create_dataset(
                    "cell_coords", data=stellar_cell_coords, compression="lzf"
                )

                data_grp.create_dataset("nb_stellar_cells", data=nb_stellar_cells)

                for iatt, (att_set) in enumerate(att_sets):
                    att_name = att_set.name

                    data_grp.create_dataset(
                        f"fesc_{att_name:s}", data=fesc_grid[iatt], compression="lzf"
                    )
                    data_grp.create_dataset(
                        f"dust_tau_{att_name:s}",
                        data=dust_tau_grid[iatt],
                        compression="lzf",
                    )


"""
Main body
"""


if __name__ == "__main__":
    Arg_parser = argparse.ArgumentParser(
        "Associate stars and halos in full simulattion"
    )

    Arg_parser.add_argument("out_nb", type=int, help="output number")

    Arg_parser.add_argument("--sub_nb", type=int, help="sub box number", default=None)

    Arg_parser.add_argument(
        "--star_sub_l",
        type=int,
        help="side length of star file sub boxes",
        default=2048,
    )

    Arg_parser.add_argument(
        "--overd_fact",
        type=float,
        help="overdensity threshold for integration radius",
        default=50,
    )

    Arg_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files?",
        default=False,
    )

    args = Arg_parser.parse_args()

    compute_stellar_Tr(**vars(args))
