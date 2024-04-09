import numpy as np
import matplotlib.pyplot as plt

# from scipy.stats import binned_statistic

from halo_properties.utils.utils import gather_h5py_files
from halo_properties.utils.output_paths import gen_paths, dataset
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from halo_properties.dust.att_coefs import get_dust_att_keys

from plot_functions.UV.uvlf.uvlfs import uvlf_plot, plot_constraints, make_uvlf
from plot_functions.generic.plot_functions import setup_plotting

import os
import h5py


def load_mags(sim_name, out_nb, dset, ext_key):
    out, assoc_out, analy_out, suffix = gen_paths(sim_name, out_nb, dset)

    print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=[ext_key])
    # except OSError as e:
    # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
    # print(e)

    return datas[ext_key][()]


setup_plotting()

# out_nbs=[14,23,34,42,52,65,82]
out_nbs = [34, 42, 52, 65, 82, 106]
overwrite = False
# stat_mthd = 'count'
lls = [0.2]  # [0.1]
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthds = ["stellar_peak"]
# assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
r200s = [1.0]
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]

# ext_keys = [k for k in get_dust_att_keys() if "LMCavg_20" in k]
# ext_keys = [k for k in get_dust_att_keys() if "LMC2_10" in k]
ext_keys = [k for k in get_dust_att_keys() if "WD_MW_5.5A_30_D03" in k]
ext_keys += ["no_dust"]
mag_ext_keys = ["mag_" + k for k in ext_keys]
cleans = [True]
mps = [True]

# print(mag_ext_keys)


nbins = 40
mag_bins = np.linspace(-25, -5, nbins)


ncols = 6
nrows = 1
nplots = len(out_nbs)
while ncols > 4:
    ncols = int(np.ceil(nplots / float(nrows)))

    nrows += 1

nrows -= 1

redshifts = np.zeros(nplots)
legends = []

fig, axs = plt.subplots(
    nrows, ncols, sharex=True, sharey=True, figsize=(27, 12), dpi=200
)
plt.subplots_adjust(hspace=0.0, wspace=0.0)
axs = np.ravel(axs)

for iplot, out_nb in enumerate(out_nbs):
    # if out_nb==106:overwrite=True

    print(f"fetching results {out_nb:d}")

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
    redshifts[iplot] = redshift

    assert (
        len(r200s) == len(assoc_mthds) == len(lls)
    ), "check input parameter lists' lengths"

    ax = axs[iplot]

    for assoc_mthd, ll, r200, mp, clean in zip(assoc_mthds, lls, r200s, mps, cleans):
        line = None

        dset = dataset(rtwo=r200, ll=ll, assoc_mthd=assoc_mthd, clean=clean, mp=mp)

        for ext_key, mag_ext_key in zip(ext_keys, mag_ext_keys):
            out_path = os.path.join("./files")
            if not os.path.isdir(out_path):
                os.makedirs(out_path)

            out_file = os.path.join(
                out_path,
                f"uvlfs_{out_nb:d}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{ext_key:s}",
            )

            if mp:
                out_file += "_mp"
            if clean:
                out_file += "_clean"

            exists = os.path.isfile(out_file)

            if overwrite or not exists:
                mags = load_mags(
                    "CoDaIII",
                    out_nb,
                    dset,
                    mag_ext_key,
                )

                print("loaded data")

                # print(mags.min(), mags.max(), np.mean(mags))

                bins, uvlf, err = make_uvlf(mags, mag_bins, Lco)

                print("made uvlf")

                with h5py.File(out_file, "w") as dest:
                    dest.create_dataset("xbins", data=bins, dtype="f4")
                    dest.create_dataset("uvlf", data=uvlf, dtype="f8")
                    dest.create_dataset("err", data=err, dtype="f8")

            else:
                with h5py.File(out_file, "r") as dest:
                    bins = dest["xbins"][()]
                    uvlf = dest["uvlf"][()]
                    err = dest["err"][()]

            # print(list(zip(bins, uvlf, err)))
            # print()
            # print("***moving everything to faint by 0.25***")

            # kern = [0.5, 0.5]
            # bins = np.convolve(bins, kern, mode="valid")
            # uvlf = np.convolve(uvlf, kern, mode="valid")
            # err = np.convolve(err, kern, mode="valid")
            # print(len(bins), len(uvlf), len(err))

            # line = uvlf_plot(fig, ax, bins + 0.25, uvlf, yerrs=err, linewidth=3, elinewidth=1, capsize=3)
            if line == None:
                line = uvlf_plot(
                    fig,
                    ax,
                    bins,
                    uvlf,
                    linewidth=3,
                    elinewidth=1,
                    capsize=3,
                    marker=None,
                )
                area = ax.fill_between(
                    bins,
                    uvlf - err,
                    uvlf + err,
                    alpha=0.5,
                    color=line.get_color(),
                )
                lines += [(line, area)]
                labels += ["CoDa III"]
            else:
                uvlf_plot(
                    fig,
                    ax,
                    bins,
                    uvlf,
                    linewidth=3,
                    elinewidth=1,
                    capsize=3,
                    marker=None,
                    ls="--",
                    color=line.get_color(),
                )

            if iplot % ncols != 0:
                ax.set_ylabel("")
            if iplot / ncols < 1 and nrows > 1:
                ax.set_xlabel("")

            # label = mag_ext_key
            label = "CoDa III"

    obs_lines, obs_labels = plot_constraints(fig, ax, redshift, prec_zed=0.5)

    leg = ax.legend(
        lines + obs_lines,
        labels + obs_labels,
        framealpha=0.0,
        title=f"z={redshift:.1f}",
        prop={"size": 14},
        loc="lower left",
    )

    legends.append(leg)

ax.set_ylim(4e-7, 1)
ax.set_xlim(-5, -23)

ax.plot([bins[4], bins[5]], [0.1, 0.1], lw=2, c="k")
ax.text(0.5 * (bins[4] + bins[5]), 0.13, "Bin width", size=12, ha="center")
print([bins[4], bins[5]])

# if nplots%2==0:
#     ax.invert_xaxis()


for idel in range(iplot + 1, nrows * ncols):
    axs[idel].remove()

# fig.suptitle(f"{assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200")


fig_name = f"./figs/uvlf_comparison_panel_prod"
fig.savefig(fig_name)
fig.savefig(fig_name + ".pdf", format="pdf")

ax.set_xlim(-13, -24)

fig_name += "_bright_end"
fig.savefig(fig_name)
fig.savefig(fig_name + ".pdf", format="pdf")


for iax, (zed, ax) in enumerate(zip(redshifts, axs)):
    # Save just the portion _inside_ the second axis's boundaries
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # extent = ax.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())

    title = legends[iax].get_title()
    legends[iax].set_title(title.get_text(), prop={"size": 15})

    ax.set_visible(True)

    ax.set_ylabel(r"$\Phi$ [Mpc$^{-3}$ mag$^{-1}$]", size=15)
    ax.set_xlabel(r"$M_{\rm AB1600}$", size=15)

    ax.tick_params(
        which="major", direction="in", labelleft=True, labelbottom=True, size=15
    )
    ax.tick_params(which="minor", direction="in", left=False, right=False)

    # panel_fig, panel_ax = make_figure(figsize_key="col_width")
    # panel_fig._axstack.add(panel_fig._make_key(ax), ax)
    for iaxx, axx in enumerate(axs):
        if iaxx != iax:
            axx.set_visible(False)

    # Pad the saved area by 10% in the x-direction and 20% in the y-direction
    fig.savefig(
        "figs/uvlf_prod_%.1f.png" % zed, bbox_inches=extent.expanded(1.275, 1.275)
    )
    fig.savefig(
        "figs/uvlf_prod_%.1f.pdf" % zed,
        bbox_inches=extent.expanded(1.275, 1.275),
        format="pdf",
    )

    # fig.savefig('figs/uvlf_prod_%.1f.png'%zed)
    # fig.savefig('figs/uvlf_prod_%.1f.pdf'%zed, format = "pdf")
