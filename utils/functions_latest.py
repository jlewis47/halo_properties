"""
Contains functions for sq integrattion programmes
"""

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# from matplotlib.colors import LogNorm
# import matplotlib.patches as pat
# import matplotlib.ticker as plticker
import os

# from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

# from scipy import spatial
from halo_properties.association.read_assoc_latest import *
from matplotlib.ticker import (
    AutoMinorLocator,
    MultipleLocator,
    LogLocator,
    LinearLocator,
)
from ..params.params import *

# import sys
# from datetime import date
# from shutil import copyfile
# import filecmp
import matplotlib.patches as pat

# from scipy.interpolate import interp2d


def rmv_inf(data):
    return data[~np.inf(data)]


def linr(x, a, b):
    return x * a + b


def get_Mp(om_m, om_b, H0, Lco, Np_tot):
    return (
        3.0
        * (H0 * 1e3 / (pc * 1e6)) ** 2.0
        * (om_m - om_b)
        * (Lco * pc * 1e6) ** 3.0
        / Np_tot
        / 8.0
        / np.pi
        / G
        / Msol
    )  # En Msol 4.04*1e5)


def get_infos(info_path, out_nb, whole_side):
    """
    use pandas if possible and move to ../files
    """

    if sixdigits:
        name = "info_%06i" % out_nb
    else:
        name = "info_%05i" % out_nb

    # Get scale factor and co
    info = np.genfromtxt(
        os.path.join(info_path, name + ".txt"), skip_header=8, max_rows=10, usecols=2
    )
    t = info[0]
    a = info[1]
    H0 = info[2]
    om_m = info[3]
    om_l = info[4]
    om_k = info[5]
    om_b = info[6]
    unit_l, unit_d, unit_t = info[7:10]
    l = unit_l / pc / 1e6 / 100.0 * H0 / 100.0 / a
    Lco = l / H0 * 100  # Mpc comobile
    L = Lco * a
    px_to_m = L * pc * 1e6 / whole_side

    return (
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
    )


def get_sample_coords(shape, ndilate):
    return np.mgrid[: shape[0] * ndilate, : shape[1] * ndilate, : shape[2] * ndilate]


def sum_sph(fields, r0, dilate_coords, ndilate, weights=None):
    if np.any(weights == None):
        weights = [1 for field in fields]

    sh = fields.shape

    x, y, z = dilate_coords / float(ndilate)

    in_sph = np.linalg.norm([x - r0, y - r0, z - r0], axis=0, ord=2) <= r0

    x = np.int16(x[in_sph])
    y = np.int16(y[in_sph])
    z = np.int16(z[in_sph])

    if len(sh) == 4:
        resampled = [
            (field * weight)[x, y, z].sum() / ndilate**3
            for field, weight in zip(fields, weights)
        ]
    else:
        resampled = (fields * weights)[x, y, z].sum() / ndilate**3

    return resampled


def mean_sph(fields, r0, dilate_coords, ndilate, weights=None):
    sh = fields.shape
    ncells = 1.0
    if len(sh) == 3:
        ncells = np.prod(sh)
    else:
        ncells = np.prod(fields[0])

    return sum_sph(fields, r0, dilate_coords, ndilate, weights=None) / float(ncells)


def get_r200_ticks(r, Mp, Np_tot=4096**3, size=4096):
    """
    Return mass of a halo based on r200 (in cell units)
    """

    out = ((r / size) ** 3.0) * Mp * 800 * np.pi * Np_tot / 3.0

    return out


def ax_setup(axis, extremax=plt.xlim(), extremay=plt.ylim(), ylog=True, xlog=True):
    """
    Duplicate axes
    Setup minor ticks
    Set fi limits
    Turn on grid
    """

    axis.set_ylim(extremay)
    axis.set_xlim(extremax)

    sizey = extremay[1] - extremay[0]
    sizex = extremax[1] - extremax[0]

    if sizey < 1:
        sizey = np.log10(1 / np.min([extremay[1], extremay[0]]))

    if sizex < 1:
        sizex = np.log10(1 / np.min([extremax[1], extremax[0]]))

    # print(sizex,sizey)

    if ylog:
        axis.yaxis.set_minor_locator(
            LogLocator(base=10, subs=[2, 4, 6, 8], numticks=(sizey) * 4)
        )
        # axis.yaxis.set_major_locator(LogLocator(base=10,numticks=sizey))
    else:
        # print(np.log10(extremay[1]/extremay[0]))
        axis.yaxis.set_minor_locator(
            AutoMinorLocator(np.floor(np.log10(extremay[1] / extremay[0])) * 4)
        )
        # axis.yaxis.set_major_locator(MultipleLocator(np.floor(np.log10(extremay[1]/extremay[0]))))

    axis.xaxis.set_minor_locator(
        LogLocator(base=10, subs=[2, 4, 6, 8], numticks=(sizex) * 4)
    )
    # axis.xaxis.set_major_locator(LogLocator(base=10,numticks=sizex))

    if xlog:
        axis.xaxis.set_minor_locator(
            LogLocator(base=10, subs=[2, 4, 6, 8], numticks=(sizex) * 4)
        )
        # axis.yaxis.set_major_locator(LogLocator(base=10,numticks=sizey))
    else:
        # print(np.log10(extremay[1]/extremay[0]))
        axis.xaxis.set_minor_locator(
            AutoMinorLocator(np.floor(np.log10(extremax[1] / extremax[0])) * 4)
        )
        # axis.yaxis.set_major_locator(MultipleLocator(np.floor(np.log10(extremay[1]/extremay[0]))))

    axis.grid(linestyle=":", linewidth=1, color="gray")

    axis.tick_params(
        axis="both", which="both", directtion="in", right="on", top="on", pad=8
    )

    return ()


def get_pos(arr):
    """
    get array N,3 posittion array
    """

    x, y, z = arr["x"], arr["y"], arr["z"]

    return np.transpose([x, y, z])


def get_std(data_set, weights=None):
    """
    Return std using numpy weighted averages
    Assumes 1d input
    """

    if weights == None:
        weights = np.ones_like(data_set)
    mx2 = np.average(data_set, weights=weights) ** 2
    x2m = np.average(data_set**2, weights=weights)
    return (x2m - mx2) ** 0.5


def do_half_round(nb):
    """
    Set coordinates to nearest pixel centre
    """
    return np.floor(nb) + 0.5


# def get_r200(M, Np_tot=4096.0**3, size=4096.0):
#     """
#     Return r200 (in cell units) based on number of particles in a halo
#     M is mass in number of DM particles
#     """

#     out = (M / Np_tot * 3.0 / 800.0 / np.pi) ** (1 / 3.0) * size
#     if out < 1.5:
#         out = 1.5
#     return out


def get_r200(M):
    """
    M in number of dark matter particle
    Return r200 (in cell units) based on number of particles in a halo
    assuming Ncell = Npart_DM and all DM particles same mass
    """

    return (M * 3.0 / 800.0 / np.pi) ** (1 / 3.0)


# def get_mask(size, ctr, r200):
# """
# From radius and size, give spherical mask centered on ctr
# """
#
# vect = np.arange(0, size)
# xvect = np.abs(vect)
# yvect = np.abs(vect[:, np.newaxis])
# zvect = np.abs(yvect[:, :, np.newaxis])
#
# return (
# np.sqrt((xvect - ctr[0]) ** 2 + (yvect - ctr[1]) ** 2 + (zvect - ctr[2]) ** 2)
# > r200
# )
#
#
# def get_clock(time):
# hh = int(time / 3600)
# mm = int(time / 60) - 60 * hh
# ss = int(time) - mm * 60 - hh * 3600
#
# return [str(hh) + ":" + str(mm) + ":" + str(ss)]
#
#
# def get_side_surface(r_px, px_to_m):
# """
# Create surface matrix for one side of the cube
# """
#
# r_px = r_px
# side = r_px * 2
# surfaces = (
# np.ones((side, side)) * (px_to_m) ** 2
# )  # Initialise all pxs of a side as full pxs
#
# return surfaces
