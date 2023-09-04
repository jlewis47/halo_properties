import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from halo_properties.utils.utils import (
    gather_h5py_files,
    ll_to_fof_suffix,
    get_r200_suffix,
    get_suffix,
    get_frad_suffix,
)
from halo_properties.utils.output_paths import gen_paths
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.generic.stat import xy_stat_usual
from plot_functions.generic.plot_functions import make_figure, xy_plot_stat
from plot_functions.hmsmr.hmsmrs import plot_hmsmr_constraints, plot_dustier_hmsmr
import os
import h5py


def load_masses(sim_name, out_nb, assoc_mthd, ll, rtwo_fact, frad, mp, clean):
    print(
        f"LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, rfesc={frad:f}, association: {assoc_mthd:s} mp={mp}"
    )

    keys = ["mass", "Mst"]

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    frad_suffix = get_frad_suffix(frad)
    suffix = get_suffix(
        fof_suffix=fof_suffix,
        rtwo_suffix=rtwo_suffix,
        frad_suffix=frad_suffix,
        mp=mp,
    )

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    if clean:
        analy_out += "_clean"

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
    # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, associatfiles.: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
    # print(e)

    return (datas["mass"], datas["Mst"])


out_nb = 52
overwrite = False
lls = [0.2, 0.2, 0.2, 0.2][::-1]
mps = [True, False, True, True][::-1]
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthds = ["stellar_peak", "stellar_peak", "stellar_peak", "stellar_peak"][::-1]
# assoc_mthds = [
#     "stellar_peak",
#     "stellar_peak",
#     "stellar_peak",
#     "fof_ctr",
#     "stellar_peak",
# ]
r200s = [1.0, 1.0, 1.0, 0.5][::-1]
frads = [1.0, 1.0, 1.0, 2.0][::-1]
cleans = [True, False, False, False][::-1]
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

fig, ax = make_figure(figsize=(8, 8))

ncols = 6
nrows = 1


for iplot, (assoc_mthd, ll, r200, frad, mp, clean) in enumerate(
    zip(assoc_mthds, lls, r200s, frads, mps, cleans)
):
    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    mp_str = ""
    if mp:
        mp_str += "mp"
    clean_str = ""
    if clean:
        clean_str += "mp"
    out_file = os.path.join(
        out_path,
        f"hmsmr_{redshift:.1f}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{frad:.1f}_{mp_str:s}_{clean_str:s}.hdf5",
    )

    exists = os.path.isfile(out_file)

    if overwrite or not exists:
        mass, stmass = load_masses(
            "CoDaIII", out_nb, assoc_mthd, ll, r200, frad, mp, clean
        )

        bins, stats = xy_stat_usual(mass, stmass, mass_bins)

        last_bin_w_enough = np.max(np.where(stats["cnts"] > 10))
        scat_mass_cut = mass_bins[last_bin_w_enough]

        high_mass, high_stmass = (
            mass[mass > scat_mass_cut],
            stmass[mass > scat_mass_cut],
        )

        # if not exists :
        with h5py.File(out_file, "w") as dest:
            dest.create_dataset("xbins", data=bins, dtype="f4", compression="lzf")
            dest.create_dataset(
                "median", data=stats["median"], dtype="f4", compression="lzf"
            )
            dest.create_dataset(
                "mean", data=stats["mean"], dtype="f4", compression="lzf"
            )
            dest.create_dataset("p5", data=stats["p5"], dtype="f4", compression="lzf")
            dest.create_dataset("p95", data=stats["p95"], dtype="f4", compression="lzf")
            dest.create_dataset(
                "low_count_stmasses", data=high_stmass, dtype="f4", compression="lzf"
            )
            dest.create_dataset(
                "low_count_masses", data=high_mass, dtype="f4", compression="lzf"
            )

    else:
        stats = {}

        with h5py.File(out_file, "r") as dest:
            bins = dest["xbins"][()]
            stats["median"] = dest["median"][()]
            stats["mean"] = dest["mean"][()]
            stats["p5"] = dest["p5"][()]
            stats["p95"] = dest["p95"][()]
            high_stmass = dest["low_count_stmasses"][()]
            high_mass = dest["low_count_masses"][()]

    line, label = xy_plot_stat(
        fig,
        ax,
        bins,
        stats,
        high_mass,
        high_stmass,
        xlabel=xlabel,
        ylabel=ylabel,
        color=next(ax._get_lines.prop_cycler)["color"],
    )

    lines.append(line)

    label = f"{assoc_mthd:s} ll={ll:.2f} rgal={r200:.1f}Xr200 rfesc={frad:.1f}"
    if mp:
        label += " mp"
    if clean:
        label += " clean"
    labels.append(label)


# lines = plot_fct(fig, ax, masses, fescs, fesc_type, redshift)

dustier_lines = []
dustier_labels = []

dustier_lines, dustier_labels = plot_dustier_hmsmr(ax, redshift, color="k", ls="--")
labels += dustier_labels
lines += dustier_lines

cst_lines, cst_labels = plot_hmsmr_constraints(ax, redshift)
labels += cst_labels
lines += cst_lines


ax.grid()

# ax.set_ylim(1e-3, 1)
# ax.set_xlim(4e7, 1e12)
# print(list(zip(lines, labels)))

plt.legend(lines, labels, framealpha=0.0)

fig_name = f"./figs/HMSMR_comparison_{out_nb:d}_{redshift:.1f}.png"
fig.savefig(fig_name)
