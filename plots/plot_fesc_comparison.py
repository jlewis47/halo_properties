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
from plot_functions.ionising.fescs import (
    fesc_Mh_plot,
    fesc_Mst_plot,
    plot_dustier_fesc,
    plot_dustier_fesc_ms,
)
import os
import h5py


def load_fescs(
    sim_name, out_nb, assoc_mthd, ll, rtwo_fact, mp, fesc_type="gas", xkey="mass"
):
    fesc_keys = {"gas": "Tr_no_dust", "full": "Tr_kext_albedo_WD_LMC2_10"}

    fesc_key = fesc_keys[fesc_type]

    print(
        f"LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}"
    )

    keys = [xkey, fesc_key, "SFR10"]

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

    sfing = datas["SFR10"] > 0

    return (datas[xkey][sfing], datas[fesc_keys[fesc_type]][sfing])


out_nb = 52
overwrite = False
fesc_type = "gas"
x_type = "Mst"  # "Mst"
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


for iplot, (assoc_mthd, ll, r200, mp) in enumerate(zip(assoc_mthds, lls, r200s, mps)):
    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    mp_str = ""
    if mp:
        mp_str += "mp_"
    out_file = os.path.join(
        out_path,
        f"fescs_{redshift:.1f}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{mp_str:s}{fesc_type:s}_{x_type:s}.hdf5",
    )

    exists = os.path.isfile(out_file)

    if overwrite or not exists:
        mass, fesc = load_fescs(
            "CoDaIII",
            out_nb,
            assoc_mthd,
            ll,
            r200,
            mp,
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

    line = xy_plot_stat(
        fig,
        ax,
        xbins,
        stats,
        high_mass_x,
        high_mass_fesc,
        xlabel=xlabel,
        ylabel="$f_{\mathrm{esc}}$",
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
if x_type == "mass":
    dustier_lines, dustier_labels = plot_dustier_fesc(ax, redshift, fkey="fesc")
if x_type == "Mst":
    dustier_lines, dustier_labels = plot_dustier_fesc_ms(ax, redshift, fkey="fesc")

labels += dustier_labels
lines += dustier_lines


ax.set_ylim(1e-3, 1)
# ax.set_xlim(4e7, 1e12)

plt.legend(lines, labels, framealpha=0.0)

fig_name = (
    f"./figs/fesc_comparison_{out_nb:d}_{redshift:.1f}_{fesc_type:s}_{x_type:s}.png"
)
fig.savefig(fig_name)
