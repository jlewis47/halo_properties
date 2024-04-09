import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from halo_properties.utils.utils import gather_h5py_files
from halo_properties.utils.output_paths import gen_paths, dataset
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.generic.stat import xy_stat
from plot_functions.generic.plot_functions import make_figure, xy_plot_stat
from plot_functions.ionising.fescs import (
    fesc_Mh_plot,
    fesc_Mst_plot,
    plot_dustier_fesc,
    plot_dustier_fesc_ms,
)
import os
import h5py


def load_fescs(
    sim_name="",
    out_nb=-1,
    dataset=None,
    fesc_type="gas",
    xkey="mass",
    clean=False,
):
    fesc_keys = {"gas": "Tr_no_dust", "full": "Tr_kext_albedo_WD_LMC2_10"}

    fesc_key = fesc_keys[fesc_type]

    print(
        f"LOADING : {out_nb:d} ll={ll:f}, {r200:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}"
    )

    keys = [xkey, fesc_key, "SFR10"]

    out, assoc_out, analy_out, suffix = gen_paths(sim_name, out_nb, dataset)

    if clean:
        analy_out += "_clean"
    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
    # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, associatfiles.: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
    # print(e)

    sfing = datas["SFR10"] > 0

    return (datas[xkey][sfing], datas[fesc_keys[fesc_type]][sfing])


out_nb = 106
overwrite = True
fesc_type = "gas"
x_type = "mass"  # "Mst"
lls = [0.2, 0.2, 0.2, 0.2]
mps = [True, True, False, False]
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthds = ["stellar_peak", "stellar_peak", "stellar_peak", "stellar_peak"]
# assoc_mthds = [
#     "stellar_peak",
#     "stellar_peak",
#     "stellar_peak",
#     "fof_ctr",
#     "stellar_peak",
# ]
r200s = [1.0, 1.0, 1.0, 1.0]
rstars = [1.0, 1.0, 1.0, 1.0]
cleans = [False, True, True, False]
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]

mnbins = 55
if x_type == "mass":
    mass_bins = np.logspace(7.5, 12, mnbins)
    plot_fct = fesc_Mh_plot
    xlabel = "$M_{\mathrm{h}}\,[M_{\odot}]$"
elif x_type == "Mst":
    mass_bins = np.logspace(3.5, 11.5, mnbins)
    plot_fct = fesc_Mst_plot
    xlabel = "$M_{\star}\,[M_{\odot}]$"

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

labels = []
lines = []

fig, ax = make_figure()

ncols = 6
nrows = 1


for iplot, (assoc_mthd, ll, r200, rstar, mp, clean) in enumerate(
    zip(assoc_mthds, lls, r200s, rstars, mps, cleans)
):
    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    mp_str = ""
    if mp:
        mp_str += "mp_"
    clean_str = ""
    if clean:
        clean_str += "mp_"

    out_file = os.path.join(
        out_path,
        f"fescs_{redshift:.1f}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{rstar:.1f}_{mp_str:s}_{clean_str:s}_{fesc_type:s}_{x_type:s}.hdf5",
    )

    if mp:
        out_file += "_mp"
    if clean:
        out_file += "_clean"

    dset = dataset(
        rtwo=r200, ll=ll, assoc_mthd=assoc_mthd, clean=clean, mp=mp, rstar=rstar
    )

    exists = os.path.isfile(out_file)

    if overwrite or not exists:
        mass, fesc = load_fescs(
            sim_name="CoDaIII",
            out_nb=out_nb,
            dataset=dset,
            fesc_type=fesc_type,
            xkey=x_type,
        )

        xbins, counts = xy_stat(mass, fesc, xbins=mass_bins, mthd="count")
        xbins, median = xy_stat(mass, fesc, xbins=mass_bins, mthd="median")
        xbins, p5 = xy_stat(
            mass, fesc, xbins=mass_bins, mthd=lambda x: np.percentile(x, 5)
        )
        xbins, p95 = xy_stat(
            mass, fesc, xbins=mass_bins, mthd=lambda x: np.percentile(x, 95)
        )

        last_bin_w_enough = np.max(np.where(counts > 10))
        scat_mass_cut = mass_bins[last_bin_w_enough]

        high_mass_x, high_mass_fesc = (
            mass[mass > scat_mass_cut],
            fesc[mass > scat_mass_cut],
        )

        # if not exists :
        with h5py.File(out_file, "w") as dest:
            dest.create_dataset("xbins", data=xbins, dtype="f4", compression="lzf")
            dest.create_dataset("median", data=median, dtype="f4", compression="lzf")
            dest.create_dataset("p5", data=p5, dtype="f4", compression="lzf")
            dest.create_dataset("p95", data=p95, dtype="f4", compression="lzf")
            dest.create_dataset(
                "low_count_fescs", data=high_mass_fesc, dtype="f4", compression="lzf"
            )
            dest.create_dataset(
                "low_count_masses", data=high_mass_x, dtype="f4", compression="lzf"
            )

        # elif overwrite and exists:

        #     with h5py.File(out_file, 'a') as dest:
        #         f_masses = dest["xbins"]
        #         f_fescs = dest["fescs"]

        #         f_masses[...] = xbins
        #         f_fescs[...] = ystat

    else:
        with h5py.File(out_file, "r") as dest:
            xbins = dest["xbins"][()]
            median = dest["median"][()]
            p5 = dest["p5"][()]
            p95 = dest["p95"][()]
            high_mass_fesc = dest["low_count_fescs"][()]
            high_mass_x = dest["low_count_masses"][()]

    stats = {}
    stats["median"] = median
    stats["p5"] = p5
    stats["p95"] = p95

    line, label = xy_plot_stat(
        fig,
        ax,
        xbins,
        stats,
        high_mass_x,
        high_mass_fesc,
        xlabel=xlabel,
        ylabel="$f_{\mathrm{esc, g}}$",
        color=next(ax._get_lines.prop_cycler)["color"],
    )

    lines.append(line)

    label = f"{assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200"
    if rstar != 1.0:
        label += f" rstar={rstar:.1f}"
    if mp:
        label += " mp"
    if clean:
        label += " clean"
    labels.append(label)


# lines = plot_fct(fig, ax, masses, fescs, fesc_type, redshift)
dustier_lines = []
dustier_labels = []
if x_type == "mass":
    dustier_lines, dustier_labels = plot_dustier_fesc(
        ax, redshift, fkey="fesc", color="k"
    )
if x_type == "Mst":
    dustier_lines, dustier_labels = plot_dustier_fesc_ms(
        ax, redshift, fkey="fesc", color="k"
    )

labels += dustier_labels
lines += dustier_lines

ax.grid()

ax.set_ylim(1e-3, 1)
# ax.set_xlim(4e7, 1e12)

plt.legend(lines, labels, framealpha=0.0)

fig_name = (
    f"./figs/fesc_comparison_{out_nb:d}_{redshift:.1f}_{fesc_type:s}_{x_type:s}.png"
)
fig.savefig(fig_name)
