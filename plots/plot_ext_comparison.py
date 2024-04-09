import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from halo_properties.utils.utils import gather_h5py_files
from halo_properties.utils.output_paths import gen_paths, dataset
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.generic.stat import xy_stat
from plot_functions.generic.plot_functions import make_figure
from plot_functions.UV.extinction import make_ext, plot_ext, plot_dustier_ext
from halo_properties.dust.att_coefs import get_dust_att_keys
import os
import h5py


def load_mags(sim_name, out_nb, dset, ext_key):
    out, assoc_out, analy_out, suffix = gen_paths(sim_name, out_nb, dset)

    print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=ext_key)
    # except OSError as e:
    # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
    # print(e)

    return [datas[ext_key][()] for ext_key in ext_key]


out_nbs = [106]
overwrite = False
lls = [0.2, 0.2, 0.2, 0.2, 0.2]
mps = [True, False, False, True, True]
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthds = [
    "stellar_peak",
    "stellar_peak",
    "stellar_peak",
    "stellar_peak",
    "stellar_peak",
]
# assoc_mthds = [
#     "stellar_peak",
#     "stellar_peak",
#     "stellar_peak",
#     "fof_ctr",
#     "stellar_peak",
# ]
r200s = [1.0, 1.0, 1.0, 1.0, 2.0]
rstars = [1.0, 1.0, 1.0, 1.0, 1.0]
cleans = [True, True, False, True, False]
max_dtms = [0.5, 0.5, 0.5, 0.1, 0.5]
fig, ax = make_figure()
# ext_key = [k for k in get_dust_att_keys() if "WD_LMC2_10" in k][0]
ext_key = ["mag_" + k for k in get_dust_att_keys() if "LMCavg_20" in k][0]
nbins = 40
mag_bins = np.linspace(-25, -5, nbins)


for out_nb in out_nbs:
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

    mags = []
    exts = []
    labels = []

    fig, ax = make_figure()

    ncols = 6
    nrows = 1

for assoc_mthd, ll, r200, mp, clean, dtm_max in zip(
    assoc_mthds, lls, r200s, mps, cleans, max_dtms
):
    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    out_file = os.path.join(
        out_path,
        f"exts_{out_nb:d}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{ext_key:s}",
    )

    if dtm_max != 0.5:
        out_file += f"_dtm{dtm_max:.2f}"
    if mp:
        out_file += "_mp"
    if clean:
        out_file += "_clean"

    dset = dataset(
        rtwo=r200, ll=ll, assoc_mthd=assoc_mthd, clean=clean, mp=mp, max_DTM=dtm_max
    )

    print()

    exists = os.path.isfile(out_file)

    if overwrite or not exists:
        loc_mags = load_mags(
            "CoDaIII",
            out_nb,
            dset,
            [ext_key, "mag_no_dust"],
            # [mag_ext_key, "mag_no_dust"],
        )

        bins, ext = make_ext(loc_mags[1], loc_mags[0], mag_bins)

        with h5py.File(out_file, "w") as dest:
            dest.create_dataset("xbins", data=bins, dtype="f4")
            dest.create_dataset("ext", data=ext, dtype="f8")
            # dest.create_dataset("err", data = err, dtype='f8')

    else:
        with h5py.File(out_file, "r") as dest:
            bins = dest["xbins"][()]
            ext = dest["ext"][()]

    mags.append(bins)
    exts.append(ext)

    label = f"{assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200 max(dtm) {dtm_max:.2f}"
    if mp:
        label += " mp"
    if clean:
        label += " clean"
    labels.append(label)

    lines = plot_ext(fig, ax, mags, exts)

    # dustier_lines, dustier_labels = plot_dustier_fesc(ax, redshift, fkey="fesc")

    # labels += dustier_labels
    # lines += dustier_lines

dst_line, dst_label = plot_dustier_ext(ax, redshift=redshift, color="k")


plt.legend(lines + dst_line, labels + dst_label, framealpha=0.0)

fig_name = f"./figs/ext_comparison_{out_nb:d}"
fig.savefig(fig_name)
