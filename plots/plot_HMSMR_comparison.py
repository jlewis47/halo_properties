import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from halo_properties.utils.utils import (
    gather_h5py_files,
    ll_to_fof_suffix,
    get_r200_suffix,
    get_suffix,
)
from halo_properties.utils.output_paths import gen_paths
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.generic.stat import xy_stat
from plot_functions.generic.plot_functions import make_figure, xy_plot_stat
from plot_functions.hmsmr.hmsmrs import plot_hmsmr_constraints, plot_dustier_hmsmr
import os
import h5py


def load_masses(sim_name, out_nb, assoc_mthd, ll, rtwo_fact, mp):
    print(
        f"LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s} mp={mp}"
    )

    keys = ["mass", "Mst"]

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix=fof_suffix, rtwo_suffix=rtwo_suffix, mp=mp)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
    # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, associatfiles.: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
    # print(e)

    return (datas["mass"], datas["Mst"])


out_nb = 52
overwrite = False
lls = [0.2, 0.2]
mps = [False, True]
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthds = ["stellar_peak", "stellar_peak"]
# assoc_mthds = [
#     "stellar_peak",
#     "stellar_peak",
#     "stellar_peak",
#     "fof_ctr",
#     "stellar_peak",
# ]
r200s = [1.0, 1.0]
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]

mnbins = 55
mass_bins = np.logspace(7.5, 12, mnbins)
xlabel = "$M_{\mathrm{h}}\,[M_{\odot}]$"
ylabel = "$M_{\star}\,[M_{\odot}]$"

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


for iplot, (assoc_mthd, ll, r200, mp) in enumerate(zip(assoc_mthds, lls, r200s, mps)):
    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    mp_str = ""
    if mp:
        mp_str += "mp"
    out_file = os.path.join(
        out_path,
        f"hmsmr_{redshift:.1f}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{mp_str:s}.hdf5",
    )

    exists = os.path.isfile(out_file)

    if overwrite or not exists:
        mass, stmass = load_masses("CoDaIII", out_nb, assoc_mthd, ll, r200, mp)
        xbins, counts = xy_stat(mass, stmass, xbins=mass_bins, mthd="count")
        xbins, median = xy_stat(mass, stmass, xbins=mass_bins, mthd="median")
        xbins, mean = xy_stat(mass, stmass, xbins=mass_bins, mthd="mean")
        xbins, p5 = xy_stat(
            mass, stmass, xbins=mass_bins, mthd=lambda x: np.percentile(x, 5)
        )
        xbins, p95 = xy_stat(
            mass, stmass, xbins=mass_bins, mthd=lambda x: np.percentile(x, 95)
        )

        last_bin_w_enough = np.max(np.where(counts > 10))
        scat_mass_cut = mass_bins[last_bin_w_enough]

        high_mass, high_stmass = (
            mass[mass > scat_mass_cut],
            stmass[mass > scat_mass_cut],
        )

        # if not exists :
        with h5py.File(out_file, "w") as dest:
            dest.create_dataset("xbins", data=xbins, dtype="f4", compression="lzf")
            dest.create_dataset("median", data=median, dtype="f4", compression="lzf")
            dest.create_dataset("mean", data=mean, dtype="f4", compression="lzf")
            dest.create_dataset("p5", data=p5, dtype="f4", compression="lzf")
            dest.create_dataset("p95", data=p95, dtype="f4", compression="lzf")
            dest.create_dataset(
                "low_count_stmasses", data=high_stmass, dtype="f4", compression="lzf"
            )
            dest.create_dataset(
                "low_count_masses", data=high_mass, dtype="f4", compression="lzf"
            )

    else:
        with h5py.File(out_file, "r") as dest:
            xbins = dest["xbins"][()]
            median = dest["median"][()]
            mean = dest["mean"][()]
            p5 = dest["p5"][()]
            p95 = dest["p95"][()]
            high_stmass = dest["low_count_stmasses"][()]
            high_mass = dest["low_count_masses"][()]

    stats = {}
    stats["median"] = median
    stats["mean"] = mean
    stats["p5"] = p5
    stats["p95"] = p95

    line = xy_plot_stat(
        fig,
        ax,
        xbins,
        stats,
        high_mass,
        high_stmass,
        xlabel=xlabel,
        ylabel=ylabel,
        color=next(ax._get_lines.prop_cycler)["color"],
    )

    lines.append(line)

    label = f"{assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200"
    if mp:
        label += " mp"
    labels.append(label)


# lines = plot_fct(fig, ax, masses, fescs, fesc_type, redshift)

dustier_lines = []
dustier_labels = []

dustier_lines, dustier_labels = plot_dustier_hmsmr(ax, redshift)
labels += dustier_labels
lines += dustier_lines

cst_lines, cst_labels = plot_hmsmr_constraints(ax, redshift)
labels += cst_labels
lines += cst_lines


# ax.set_ylim(1e-3, 1)
# ax.set_xlim(4e7, 1e12)

plt.legend(lines, labels, framealpha=0.0)

fig_name = f"./figs/HMSMR_comparison_{out_nb:d}_{redshift:.1f}.png"
fig.savefig(fig_name)
