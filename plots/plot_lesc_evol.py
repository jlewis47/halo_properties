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
from plot_functions.generic.plot_functions import make_figure, setup_plotting
from plot_functions.ionising.lescs import lesc_Mh_plot, plot_dustier_lesc
import os
import h5py


def load_data(
    sim_name, out_nb, assoc_mthd, ll, rtwo_fact, fesc_type="gas", xkey="mass"
):
    fesc_keys = {"gas": "Tr_no_dust", "full": "Tr_kext_albedo_WD_LMC2_10"}

    fesc_key = fesc_keys[fesc_type]

    print(
        f"LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}"
    )

    keys = [xkey, fesc_key, "SFR10"]

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix, rtwo_suffix)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=keys + ["Lintr"])
    # except OSError as e:
    # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, associatfiles.: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
    # print(e)

    sfing = datas["SFR10"] > 0

    return (datas[xkey], datas[fesc_key], datas["Lintr"], sfing)


setup_plotting()

out_nbs = [23, 34, 42, 52, 65, 82, 106]
# out_nbs = [106]

overwrite = False
clean = True
fesc_type = "gas"
x_type = "mass"
stat_mthd = "mean"
ll = 0.2
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthd = "stellar_peak"
# assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
r200 = 1.0
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]

mnbins = 55
mass_bins = np.logspace(7.5, 12, mnbins)


masses = []
lescs = []
lescs_sfing = []
labels = []
lines = []
redshifts = []

fig, ax = make_figure()

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
    redshifts.append(redshift)

    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    out_file = os.path.join(
        out_path,
        f"lescs_{redshift:.1f}_{stat_mthd:s}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{fesc_type:s}",
    )

    exists = os.path.isfile(out_file)

    if overwrite or not exists:
        mass, fesc, lintr, sfing = load_data(
            "CoDaIII", out_nb, assoc_mthd, ll, r200, fesc_type=fesc_type, xkey="mass"
        )
        xbins, lesc = xy_stat(mass, fesc * lintr, xbins=mass_bins, mthd=stat_mthd)
        xbins, lesc_sfing = xy_stat(
            mass[sfing], (fesc * lintr)[sfing], xbins=mass_bins, mthd=stat_mthd
        )

        # if not exists :
        with h5py.File(out_file, "w") as dest:
            dest.create_dataset("xbins", data=xbins, dtype="f4", compression="lzf")
            dest.create_dataset(
                "%s_lescs" % stat_mthd, data=lesc, dtype="f8", compression="lzf"
            )
            dest.create_dataset(
                "%s_lescs_sfing" % stat_mthd,
                data=lesc_sfing,
                dtype="f8",
                compression="lzf",
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
            lesc = dest["%s_lescs" % stat_mthd][()]
            lesc_sfing = dest["%s_lescs_sfing" % stat_mthd][()]

    masses.append(xbins)
    lescs.append(lesc)
    lescs_sfing.append(lesc_sfing)

    # labels.append(f"{stat_mthd:s} {assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200")

for z, mass, lesc, lesc_sfing in zip(redshifts, masses, lescs, lescs_sfing):
    # print(mass, lesc, lesc_sfing)
    lines.append(lesc_Mh_plot(fig, ax, mass, lesc, fesc_type, redshift=None))
    labels.append(f"{z:.1f}")
    lesc_Mh_plot(
        fig,
        ax,
        mass,
        lesc_sfing,
        fesc_type,
        redshift=None,
        ls="--",
        c=lines[-1].get_color(),
    )

# dustier_lines, dustier_labels = plot_dustier_lesc(ax, redshift, fkey="fesc")

# labels += dustier_labels
# lines += dustier_lines
plt.grid()
plt.legend(lines, labels, framealpha=0.0)

fig_name = f"./figs/lesc_evolution_{stat_mthd:s}_{assoc_mthd:s}_ll={int(100*ll):d}_{int(r200):d}Xr200_{fesc_type:s}"
fig.savefig(fig_name)
