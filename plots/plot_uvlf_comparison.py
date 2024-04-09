import numpy as np
import matplotlib.pyplot as plt
from halo_properties.utils.utils import gather_h5py_files
from halo_properties.utils.output_paths import gen_paths, dataset
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.UV.uvlf.uvlfs import uvlf_plot, make_uvlf, plot_constraints
from halo_properties.dust.att_coefs import get_dust_att_keys
from plot_functions.generic.plot_functions import make_figure

import os
import h5py


def load_mags(sim_name, out_nb, dset, ext_key=None):
    if ext_key == None:
        ext_key = "mag_no_dust"

    out, assoc_out, analy_out, suffix = gen_paths(sim_name, out_nb, dset)

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=[ext_key])
    # except OSError as e:
    # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
    # print(e)

    return datas[ext_key][()]


out_nb = 82
overwrite = False
lls = [0.2, 0.2]  # , 0.2, 0.2, 0.2, 0.2]
# lls = [0.2, 0.2, 0.2, 0.2, 0.2]
mps = [True, False]  # , False, False, True, True]
# mps = [True, False, False, True, True]
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthds = [
    "stellar_peak",
    "stellar_peak",
    # "stellar_peak",
    # "stellar_peak",
    # "stellar_peak",
]
# assoc_mthds = [
#     "stellar_peak",
#     "stellar_peak",
#     "stellar_peak",
#     "fof_ctr",
#     "stellar_peak",
# ]
# r200s = [1.0, 1.0, 1.0, 1.0, 1.0]
r200s = [1.0, 1.0]  # , 1.0, 1.0, 1.0, 1.0]
# rstars = [1.0, 1.0, 1.0, 1.0, 1.0]
rstars = [1.0, 1.0]  # , 1.0, 1.0, 1.0, 1.0]
# cleans = [True, True, False, True, True]
cleans = [True, True]  # , True, False, True, True]
# max_dtms = [0.5, 0.5, 0.5, 0.1, 0.05]
max_dtms = [0.05, 0.10]
fig, ax = make_figure()
# ext_key = ["mag_" + k for k in get_dust_att_keys() if "WD_LMC2_10" in k][0]
ext_key = ["mag_" + k for k in get_dust_att_keys() if "LMCavg_20" in k][0]
nbins = 40
mag_bins = np.linspace(-25, -5, nbins)

# print(mag_bins)

info_path = os.path.join(sim_path, f"output_{out_nb:06d}", "group_000001")

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

assert (
    len(r200s) == len(assoc_mthds) == len(lls)
), "check input parameter lists' lengths"

xs = []
uvlfs = []
errs = []
labels = []

for assoc_mthd, ll, r200, mp, clean, dtm_max in zip(
    assoc_mthds, lls, r200s, mps, cleans, max_dtms
):
    dset = dataset(
        r200=r200,
        ll=ll,
        assoc_mthd=assoc_mthd,
        clean=clean,
        mp=mp,
        max_DTM=dtm_max,
        neb_cont_file_name=neb_cont_file_name,
    )

    # overwrite = False
    # if r200 == 2:
    #     overwrite = True

    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    out_file = os.path.join(
        out_path,
        f"uvlfs_{out_nb:d}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{ext_key:s}",
    )

    if dtm_max != 0.5:
        out_file += f"_dtm{dtm_max:.2f}"

    if mp:
        out_file += "_mp"
    if clean:
        out_file += "_clean"

    exists = os.path.isfile(out_file)

    # if dtm_max != 0.5:
    #     overwrite = True

    if overwrite or not exists:
        mags = load_mags("CoDaIII", out_nb, dset, ext_key)

        # print(mags.min(), mags.max(), np.mean(mags))

        bins, uvlf, err = make_uvlf(mags, mag_bins, Lco)

        with h5py.File(out_file, "w") as dest:
            dest.create_dataset("xbins", data=bins, dtype="f4")
            dest.create_dataset("uvlf", data=uvlf, dtype="f8")
            dest.create_dataset("uvlf_err", data=err, dtype="f8")

    else:
        with h5py.File(out_file, "r") as dest:
            bins = dest["xbins"][()]
            uvlf = dest["uvlf"][()]
            err = dest["uvlf_err"][()]

    xs.append(bins)
    uvlfs.append(uvlf)
    errs.append(err)

    # print(bins, uvlf)

    label = f"{assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200 max(dtm) {dtm_max:.2f}"
    if mp:
        label += " mp"
    if clean:
        label += " clean"
    labels.append(label)

lines = uvlf_plot(fig, ax, xs, uvlfs, redshift, yerrs=errs)
obs_lines, obs_labels = plot_constraints(fig, ax, redshift)

labels += obs_labels
lines += obs_lines

ax.legend(lines, labels, framealpha=0.0)

ax.set_ylim(4e-7, 1)
ax.set_xlim(-13, -24)

fig_name = f"./figs/uvlf_comparison_{redshift:.1f}_{out_nb:d}"
if "no_dust" not in ext_key:
    fig_name += "_ext"
fig_name += ".png"

fig.savefig(fig_name)
