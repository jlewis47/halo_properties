import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

from halo_properties.utils.utils import gather_h5py_files
from halo_properties.utils.output_paths import gen_paths, dataset
from halo_properties.utils.functions_latest import get_infos
from halo_properties.params.params import *

from plot_functions.generic.stat import mass_function
from plot_functions.generic.plot_functions import make_figure
from plot_functions.smf.smfs import plot_constraints, smf_plot, make_smf


def load_stellar_masses(sim_name, out_nb, dset):
    keys = ["Mst"]

    out, assoc_out, analy_out, suffix = gen_paths(sim_name, out_nb, dset)

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
    # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
    # print(e)

    return datas["Mst"][()]


out_nb = 106
overwrite = False
lls = [0.2, 0.2, 0.2, 0.2]  # , 0.2]  # , 0.2]
mps = [True, False, True, False]  # , True]  # , True]
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthds = [
    "stellar_peak",
    "stellar_peak",
    "stellar_peak",
    "stellar_peak",
]  # , "stellar_peak"]
# "stellar_peak",
# ]
# assoc_mthds = [
#     "stellar_peak",
#     "stellar_peak",
#     "stellar_peak",
#     "fof_ctr",
#     "stellar_peak",
# ]
r200s = [1.0, 1.0, 1.0, 1.0]  # , 1.0]  # , 2.0]
rstars = [1.0, 1.0, 1.0, 1.0]  # , 1.0]  # , 1.0]
cleans = [True, True, False, False]  # , False]  # , True]
max_dtms = [0.5, 0.5, 0.5, 0.5]  # , 0.5]  # , 0.1]

fig, ax = make_figure()

nbins = 25
mass_bins = np.logspace(4.0, 11, nbins)

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

masses = []
smfs = []
labels = []
errs = []
# lines = []

for iplot, (assoc_mthd, ll, r200, rstar, mp, clean, dtm_max) in enumerate(
    zip(assoc_mthds, lls, r200s, rstars, mps, cleans, max_dtms)
):
    dset = dataset(
        r200=r200, ll=ll, assoc_mthd=assoc_mthd, clean=clean, mp=mp, max_DTM=dtm_max
    )

    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # overwrite = False
    # if dtm_max != 0.5:
    #     overwrite = True

    out_file = os.path.join(
        out_path,
        f"smfs_{out_nb:d}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}",
    )

    if dtm_max != 0.5:
        out_file += f"_dtm{dtm_max:.2f}"
    if mp:
        out_file += "_mp"
    if clean:
        out_file += "_clean"

    exists = os.path.isfile(out_file)

    # print(out_file, exists)

    if overwrite or not exists:
        stellar_masses = load_stellar_masses("CoDaIII", out_nb, dset)

        bins, smf, err = make_smf(stellar_masses, mass_bins, Lco)

        with h5py.File(out_file, "w") as dest:
            dest.create_dataset("xbins", data=bins, dtype="f4")
            dest.create_dataset("smf", data=smf, dtype="f8")
            dest.create_dataset("err", data=err, dtype="f8")

    else:
        with h5py.File(out_file, "r") as dest:
            bins = dest["xbins"][()]
            smf = dest["smf"][()]
            err = dest["err"][()]

    print(smf)

    masses.append(bins)
    smfs.append(smf)
    errs.append(err)
    label = f"{assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200"
    if rstar != 1.0:
        label += f" rstar={rstar:.1f}"
    if mp:
        label += " mp"
    if clean:
        label += " clean"
    labels.append(label)


# print(masses, smfs)
lines = smf_plot(fig, ax, masses, smfs, yerrs=errs)

cst_lines, cst_labels = plot_constraints(fig, ax, redshift)

labels += cst_labels
lines += cst_lines

ax.legend(lines, labels, framealpha=0.0)


fig.savefig(f"./figs/smf_comparison_{out_nb:d}", bbox_inches="tight")


ax.set_xlim(1e7, 1e11)
ax.set_ylim(4e-6, 2e-1)
fig.savefig(f"./figs/smf_comparison_high_mass_{out_nb:d}", bbox_inches="tight")
