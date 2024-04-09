import numpy as np
import matplotlib.pyplot as plt

from halo_properties.utils.utils import gather_h5py_files
from halo_properties.utils.output_paths import gen_paths, dataset
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from halo_properties.dust.att_coefs import get_dust_att_keys

from plot_functions.UV.UVslope.betaVSmag import (
    make_magbeta,
    plot_magbeta,
    plot_magbeta_constraints,
    plot_dustier_magbeta,
)

import os
import h5py


def load_magbeta(sim_name, out_nb, dset, ext_key):
    out, assoc_out, analy_out, suffix = gen_paths(sim_name, out_nb, dset)

    # print(analy_out)

    keys = ["mag_" + ext_key, "betas_" + ext_key]

    # try:
    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
    # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
    # print(e)

    return [datas[k][()] for k in keys]


out_nbs = [34, 42, 52, 65, 82, 106]
overwrite = False
# stat_mthd = 'count'
lls = [0.2]  # [0.1]
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthds = ["stellar_peak"]
# assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
r200s = [1.0]
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]
mps = [True]
cleans = [True]

ext_keys = get_dust_att_keys()
mag_ext_keys = ["mag_" + k for k in ext_keys]


nbins = 40
mag_bins = np.linspace(-25, -5, nbins)


ncols = 6
nrows = 1
nplots = len(out_nbs)
while ncols > 4:
    ncols = int(np.ceil(nplots / float(nrows)))

    nrows += 1

nrows -= 1

fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(27, 12))
axs = np.ravel(axs)

for iplot, out_nb in enumerate(out_nbs):
    lines = []
    labels = []

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

    ax = axs[iplot]

    for assoc_mthd, ll, r200, mp, clean in zip(assoc_mthds, lls, r200s, mps, cleans):
        dset = dataset(rtwo=r200, ll=ll, assoc_mthd=assoc_mthd, clean=clean, mp=mp)

        for ext_key, mag_ext_key in zip(ext_keys, mag_ext_keys):
            out_path = os.path.join("./files")
            if not os.path.isdir(out_path):
                os.makedirs(out_path)

            out_file = os.path.join(
                out_path,
                f"magbeta_{out_nb:d}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{ext_key:s}",
            )
            if mp:
                out_file += "_mp"
            if clean:
                out_file += "_clean"

            exists = os.path.isfile(out_file)

            if overwrite or not exists:
                mags, betas = load_magbeta("CoDaIII", out_nb, dset, ext_key)

                # print(mags.min(), mags.max(), np.mean(mags))

                bins, rel = make_magbeta(mags, betas, mag_bins)

                with h5py.File(out_file, "w") as dest:
                    dest.create_dataset("mags", data=bins, dtype="f4")
                    dest.create_dataset("magVbeta", data=rel, dtype="f4")

            else:
                with h5py.File(out_file, "r") as dest:
                    bins = dest["mags"][()]
                    rel = dest["magVbeta"][()]

            # print(bins, mag_bins[:-1], np.diff(mag_bins)*0.5)

            # print(list(zip(bins, uvlf, err)))
            # print()
            # print(list(zip(bins,rel)))
            # fig, ax, bins + 2.5 * np.log10(1.0 / 0.8), rel, linewidth=3
            line = plot_magbeta(fig, ax, bins, rel, linewidth=3)
            label = mag_ext_key

            lines += [line]
            labels += [label]

    obs_lines, obs_labels = plot_magbeta_constraints(ax, redshift, bins)

    dustier_line, dustier_label = plot_dustier_magbeta(
        ax, redshift, zprec=0.1, color="tab:purple"
    )

    # print(labels, obs_labels)
    # print(lines, obs_lines)

    ax.tick_params(labelleft=True, labelbottom=True)

    ax.legend(
        lines + obs_lines + dustier_line,
        labels + obs_labels + dustier_label,
        framealpha=0.0,
        title=f"z={redshift:.1f}",
    )

ax.set_ylim(-3, -1)
ax.set_xlim(-15, -23)

if nplots % 2 != 0:
    ax.invert_xaxis()

# if nplots%2==0:
#     ax.invert_xaxis()


for idel in range(iplot + 1, nrows * ncols):
    axs[idel].remove()


fig.suptitle(f"{assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200")


fig_name = f"./figs/magbeta_comparison_panel_ext_comp"
fig.savefig(fig_name)
