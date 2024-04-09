import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.stats import gaussian_kde, binned_statistic
from plot_functions.generic.plot_functions import make_figure
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def gather_stat_files(path, tgt_mass, out_nb="none", nb_files="none"):
    if out_nb != "none":
        files = [
            f
            for f in os.listdir(path)
            if "stellar_pties" in f and tgt_mass in f and "%04i" % (out_nb) in f
        ]
    else:
        files = [f for f in os.listdir(path) if "stellar_pties" in f and tgt_mass in f]

    if nb_files != "none":
        files = files[:nb_files]

    # print([f for f in os.listdir(path) if "stellar_pties" in f and tgt_mass in f],"%04i"%(out_nb), [f for f in os.listdir(path) if "stellar_pties" in f and tgt_mass in f and "%04i"%(out_nb) in f])

    # print(files, files[:1])

    with h5py.File(os.path.join(path, files[0]), "r") as src:
        keys = list(src.keys())

    output = [[] for k in keys]

    for f in files:
        with h5py.File(os.path.join(path, f), "r") as src:
            for ik, k in enumerate(keys):
                output[ik].extend(src[k][()])

    out_dict = {k: np.array(v) for k, v in zip(keys, output)}

    return (keys, out_dict)


nbins = 20
tgt_mass = 1e11
tgt_mass_str = ("%.1e" % tgt_mass).replace(".", "p")
print(tgt_mass_str)
zed = 6.0


dustier_out_nb_to_zed = {27: 10.0, 32: 9.0, 37: 8.0, 42: 7.0, 47: 6.0, 52: 5.0}
dustier_zed_to_out_nb = dict((v, k) for k, v in dustier_out_nb_to_zed.items())
dustier_path = "/ccs/home/jlewis/plot_functions/constraints/dustier_stel_pties"
dustier_keys, dustier_datas = gather_stat_files(
    dustier_path, tgt_mass_str, out_nb=dustier_zed_to_out_nb[zed]
)

codaiii_out_nb_to_zed = {
    14: 15.0,
    23: 12.0,
    34: 10.0,
    42: 10.0,
    52: 8.0,
    62: 7.0,
    82: 6.0,
    106: 5.0,
}
codaiii_zed_to_out_nb = dict((v, k) for k, v in codaiii_out_nb_to_zed.items())
codaiii_path = "/lustre/orion/proj-shared/ast031/jlewis/CoDaIII_analysis/results_stellar_peak_halos_82_ll_0p200"
codaiii_keys, codaiii_datas = gather_stat_files(
    codaiii_path, tgt_mass_str, nb_files=512
)

print(dustier_keys, codaiii_keys)


key_translation = {
    "Z": "stellar Z",
    "mass": "stellar mass",
    #  "lintr":"stellar ionizing luminosity",
    "lintr": "halo lintr",
    "radius": "stellar radii",
    "age": "stellar age",
    "beta": "stellar beta no_dust",
    "beta ext_LMC": "stellar beta kext_albedo_WD_LMC2_10",
    "gas density": "gas density",
    "dust": "dust density",
    "fesc": "stellar fesc no_dust",
    "fesc ext_LMC": "stellar fesc kext_albedo_WD_LMC2_10",
    "mag": "stellar mag no_dust",
    "mag ext_LMC": "stellar mag kext_albedo_WD_LMC2_10",
    "xion": "gas xion",
}

codaiii_st_mass_key_ind = np.argwhere([k == "stellar mass" for k in codaiii_keys])[0][0]
dustier_st_mass_key_ind = np.argwhere([k == "mass" for k in dustier_keys])[0][0]

for key in dustier_keys:
    print(key)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # print(key,data)

    codaiii_key = key_translation[key]
    # print(codaiii_key, codaiii_keys)
    # codaiii_key_ind = np.argwhere([k==codaiii_key for k in codaiii_keys])[0][0]
    # print(codaiii_key, codaiii_key_ind)
    codaiii_data = codaiii_datas[codaiii_key]

    data = np.asarray(dustier_datas[key])
    codaiii_data = np.asarray(codaiii_data)

    # print(codaiii_data)
    l_codaiii = codaiii_data.shape
    l_dustier = data.shape

    # tirage
    # sample_inds = np.int32(np.random.uniform(low=0,high=l_codaiii-1,size=l_dustier))
    # codaiii_data = np.asarray(codaiii_data)[sample_inds]

    codaiii_mean = np.mean(codaiii_data)
    codaiii_median = np.median(codaiii_data)
    mean = np.mean(data)
    median = np.median(data)

    if "mag" in key:
        stm_mean = -2.5 * np.log10(
            np.average(10 ** (data / -2.5), weights=np.asarray(dustier_datas["mass"]))
        )
        codaiii_stm_mean = -2.5 * np.log10(
            np.average(
                10 ** (codaiii_data / -2.5),
                weights=np.asarray(codaiii_datas["stellar mass"]),
            )
        )
        # codaiii_stm_mean = -2.5*np.log10(np.average(10**(codaiii_data/-2.5), weights=np.asarray(codaiii_datas["stellar mass"])[sample_inds]))
    else:
        stm_mean = np.average(data, weights=np.asarray(dustier_datas["mass"]))
        codaiii_stm_mean = np.average(
            codaiii_data, weights=np.asarray(codaiii_datas["stellar mass"])
        )
        # codaiii_stm_mean = np.average(codaiii_data, weights=np.asarray(codaiii_datas["stellar mass"])[sample_inds])

    if "fesc" in key:
        print(np.mean(codaiii_data[codaiii_data < 1e-10]))
        data[data < 1e-10] = 1e-10
        codaiii_data[codaiii_data < 1e-10] = 1e-10
    elif "Z" in key:
        data[data < 1e-6] = 1e-6
        codaiii_data[codaiii_data < 1e-6] = 1e-6
    elif "age" in key:
        data[data < 0.1] = 0.1
    elif "dust" in key:
        data[data < 1e-30] = 1e-30
        codaiii_data[codaiii_data < 1e-30] = 1e-30
    elif "mass" in key:
        codaiii_data /= 0.8
        data /= 0.8
    elif "mag" in key:
        # codaiii_data+=2.5*np.log10(1./0.8)
        data -= 2.5 * np.log10(1.0 / 0.8)
    # elif "radius" in key:
    #     print("radii", data.mean())
    #     codaiii_data /= 1.4938015821857216
    #     print(data.mean())

    print(
        f"\tMean\tMedian\nCoDaIII {codaiii_mean:.1e}\t{codaiii_median:.1e}\nDUSTiER {mean:.1e}\t{median:.1e}"
    )

    if np.any(np.isfinite(data) == False):
        print(data)

    if not np.any(np.asarray(data) < 0):
        m = np.log10(np.min(data[data > 0])) - 0.2
        bins = np.logspace(m, np.log10(np.max(data)) + 0.2, nbins)
        ax.set_xscale("log")
    else:
        max_data = np.max(data)
        if max_data > 0:
            max_data *= 1.1
        else:
            max_data *= 0.9

        min_data = np.min(data)
        if min_data < 0:
            min_data *= 1.1
        else:
            min_data *= 0.9

        bins = np.linspace(min_data, max_data, nbins)

    hist, bins = np.histogram(data, bins=bins)
    codaiii_hist, dummy = np.histogram(codaiii_data, bins=bins)

    # kde = gaussian_kde(data)
    # kde_codaiii = gaussian_kde(codaiii_data)

    # print(kde)

    # print(kde.pdf(bins))

    # print(bins, hist, np.min(data), np.mean(data), np.max(data))
    # print(list(zip(bins,hist)), np.mean(data), np.min(data), np.max(data))

    bins = bins[:-1] + np.diff(bins) * 0.5

    # norm = kde.integrate_box_1d(bins[0], bins[-1])
    # norm_codaiii = kde_codaiii.integrate_box_1d(bins[0], bins[-1])

    plot = ax.plot(
        bins,
        codaiii_hist / np.sum(codaiii_hist),
        lw=3,
        ds="steps-mid",
        label="CoDa III",
    )
    ax.axvline(codaiii_mean, lw=3, ls=":", c=plot[0].get_color())
    ax.axvline(codaiii_stm_mean, lw=3, ls="--", c=plot[0].get_color())
    ax.axvline(codaiii_median, lw=3, ls="-", c=plot[0].get_color())
    # ax.plot(bins, kde_codaiii.pdf(bins)/norm_codaiii, lw=3, ls='--', c=plot[0].get_color())

    plot = ax.plot(bins, hist / np.sum(hist), lw=3, ds="steps-mid", label="DUSTiER")
    ax.axvline(mean, lw=3, ls=":", c=plot[0].get_color())
    ax.axvline(stm_mean, lw=3, ls="--", c=plot[0].get_color())
    ax.axvline(median, lw=3, ls="-", c=plot[0].get_color())
    # ax.plot(bins, kde.pdf(bins), lw=3, ls='--', c=plot[0].get_color())

    ax.legend(title=f"z=6 $M_h={tgt_mass:.1e}$" + "$ \, \mathrm{M_\odot}$")

    ax.set_yscale("log")

    ax.set_ylabel("PDF")
    ax.set_xlabel(f"{codaiii_key:s}")

    if "fesc" in key:
        ax.set_xlim(1e-6, 1.05)

    fig.savefig(f"./figs/{key.replace(' ','_'):s}_{tgt_mass_str:s}")


##plot the escape fraction as a function of gas density for both simulations
fig, ax = make_figure()

gas_bins = np.logspace(-30, -22, 25)
fesc_bins = np.logspace(-10, 0, 25)

divider = make_axes_locatable(ax)

cond = np.ones_like(codaiii_datas["stellar fesc no_dust"]) == 1.0
# cond = codaiii_datas["stellar fesc no_dust"] > 1e-10

# add y hist panel for both sims
axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
axHisty.set_xscale("log")

axHisty.hist(
    np.log10(codaiii_datas["stellar fesc no_dust"][cond]),
    bins=np.log10(fesc_bins),
    orientation="horizontal",
    color="tab:blue",
    alpha=0.5,
    density=True,
)
axHisty.hist(
    np.log10(dustier_datas["fesc"]),
    bins=np.log10(fesc_bins),
    orientation="horizontal",
    color="tab:orange",
    alpha=0.5,
    density=True,
)

axHisty.tick_params(axis="y", labelleft=False)

# add x hist panel for both sims
axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
axHistx.set_yscale("log")

axHistx.hist(
    np.log10(codaiii_datas["gas density"][cond]),
    bins=np.log10(gas_bins),
    color="tab:blue",
    alpha=0.5,
    density=True,
)
axHistx.hist(
    np.log10(dustier_datas["gas density"]),
    bins=np.log10(gas_bins),
    color="tab:orange",
    alpha=0.5,
    density=True,
)

axHistx.tick_params(axis="x", labelbottom=False)

codaiii_mean_fesc, bins, counts = binned_statistic(
    codaiii_datas["gas density"][cond],
    codaiii_datas["stellar fesc no_dust"][cond],
    statistic="mean",
    bins=gas_bins,
)
dustier_mean_fesc, bins, counts = binned_statistic(
    dustier_datas["gas density"], dustier_datas["fesc"], statistic="mean", bins=gas_bins
)

ax.plot(
    np.log10(bins[:-1] + np.diff(bins) * 0.5),
    np.log10(codaiii_mean_fesc),
    lw=3,
    label="CoDa III",
)
ax.plot(
    np.log10(bins[:-1] + np.diff(bins) * 0.5),
    np.log10(dustier_mean_fesc),
    lw=3,
    label="DUSTiER",
)

# compute median for both sims
codaiii_median_fesc, bins, counts = binned_statistic(
    codaiii_datas["gas density"][cond],
    codaiii_datas["stellar fesc no_dust"][cond],
    statistic="median",
    bins=gas_bins,
)
dustier_median_fesc, bins, counts = binned_statistic(
    dustier_datas["gas density"],
    dustier_datas["fesc"],
    statistic="median",
    bins=gas_bins,
)

# plot medians w dashes
ax.plot(
    np.log10(bins[:-1] + np.diff(bins) * 0.5),
    np.log10(codaiii_median_fesc),
    lw=3,
    ls="--",
    c="tab:blue",
)
ax.plot(
    np.log10(bins[:-1] + np.diff(bins) * 0.5),
    np.log10(dustier_median_fesc),
    lw=3,
    ls="--",
    c="tab:orange",
)


codaiii_fesc_gas_hist, xedges, yedges = np.histogram2d(
    codaiii_datas["gas density"][cond],
    codaiii_datas["stellar fesc no_dust"][cond],
    bins=(gas_bins, fesc_bins),
)
ax.imshow(
    codaiii_fesc_gas_hist.T,
    extent=np.log10([xedges[0], xedges[-1], yedges[0], yedges[-1]]),
    origin="lower",
    aspect="auto",
    cmap="Blues",
    alpha=0.5,
)


dustier_fesc_gas_hist, xedges, yedges = np.histogram2d(
    dustier_datas["gas density"], dustier_datas["fesc"], bins=(gas_bins, fesc_bins)
)
ax.imshow(
    dustier_fesc_gas_hist.T,
    extent=np.log10([xedges[0], xedges[-1], yedges[0], yedges[-1]]),
    origin="lower",
    aspect="auto",
    cmap="Oranges",
    alpha=0.5,
)


# ax.set_xscale("log")
# ax.set_yscale("log")

ax.set_ylabel(r"Escape fraction")
ax.set_xlabel(r"$\rho_\mathrm{gas}$, density code units")
ax.legend()

# ax.set_ylim(1e-4,1.0)

fig.savefig("./figs/fesc_vs_gas_density.png")

##plot the escape fraction as a function of gas density for both simulations
fig, ax = make_figure()

gas_bins = np.logspace(-30, -22, 25)
fesc_bins = np.logspace(-10, 0, 25)

divider = make_axes_locatable(ax)

# add y hist panel for both sims
axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)
axHisty.set_xscale("log")

axHisty.hist(
    np.log10(codaiii_datas["stellar fesc no_dust"]),
    bins=np.log10(fesc_bins),
    orientation="horizontal",
    color="tab:blue",
    alpha=0.5,
    density=True,
)
axHisty.hist(
    np.log10(dustier_datas["fesc"]),
    bins=np.log10(fesc_bins),
    orientation="horizontal",
    color="tab:orange",
    alpha=0.5,
    density=True,
)

axHisty.tick_params(axis="y", labelleft=False)

# add x hist panel for both sims
axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
axHistx.set_yscale("log")

axHistx.hist(
    np.log10(codaiii_datas["gas density"] * (1.0 - codaiii_datas["gas xion"])),
    bins=np.log10(gas_bins),
    color="tab:blue",
    alpha=0.5,
    density=True,
)
axHistx.hist(
    np.log10(dustier_datas["gas density"] * (1.0 - dustier_datas["xion"])),
    bins=np.log10(gas_bins),
    color="tab:orange",
    alpha=0.5,
    density=True,
)

axHistx.tick_params(axis="x", labelbottom=False)

codaiii_mean_fesc, bins, counts = binned_statistic(
    codaiii_datas["gas density"] * (1.0 - codaiii_datas["gas xion"]),
    codaiii_datas["stellar fesc no_dust"],
    statistic="mean",
    bins=gas_bins,
)
dustier_mean_fesc, bins, counts = binned_statistic(
    dustier_datas["gas density"] * (1.0 - dustier_datas["xion"]),
    dustier_datas["fesc"],
    statistic="mean",
    bins=gas_bins,
)

ax.plot(
    np.log10(bins[:-1] + np.diff(bins) * 0.5),
    np.log10(codaiii_mean_fesc),
    lw=3,
    label="CoDa III",
)
ax.plot(
    np.log10(bins[:-1] + np.diff(bins) * 0.5),
    np.log10(dustier_mean_fesc),
    lw=3,
    label="DUSTiER",
)

# compute median for both sims
codaiii_median_fesc, bins, counts = binned_statistic(
    codaiii_datas["gas density"] * (1.0 - codaiii_datas["gas xion"]),
    codaiii_datas["stellar fesc no_dust"],
    statistic="median",
    bins=gas_bins,
)
dustier_median_fesc, bins, counts = binned_statistic(
    dustier_datas["gas density"] * (1.0 - dustier_datas["xion"]),
    dustier_datas["fesc"],
    statistic="median",
    bins=gas_bins,
)

# plot medians w dashes
ax.plot(
    np.log10(bins[:-1] + np.diff(bins) * 0.5),
    np.log10(codaiii_median_fesc),
    lw=3,
    ls="--",
    c="tab:blue",
)
ax.plot(
    np.log10(bins[:-1] + np.diff(bins) * 0.5),
    np.log10(dustier_median_fesc),
    lw=3,
    ls="--",
    c="tab:orange",
)


codaiii_fesc_gas_hist, xedges, yedges = np.histogram2d(
    codaiii_datas["gas density"] * (1.0 - codaiii_datas["gas xion"]),
    codaiii_datas["stellar fesc no_dust"],
    bins=(gas_bins, fesc_bins),
)
ax.imshow(
    codaiii_fesc_gas_hist.T,
    extent=np.log10([xedges[0], xedges[-1], yedges[0], yedges[-1]]),
    origin="lower",
    aspect="auto",
    cmap="Blues",
    alpha=0.5,
)


dustier_fesc_gas_hist, xedges, yedges = np.histogram2d(
    dustier_datas["gas density"] * (1.0 - dustier_datas["xion"]),
    dustier_datas["fesc"],
    bins=(gas_bins, fesc_bins),
)
ax.imshow(
    dustier_fesc_gas_hist.T,
    extent=np.log10([xedges[0], xedges[-1], yedges[0], yedges[-1]]),
    origin="lower",
    aspect="auto",
    cmap="Oranges",
    alpha=0.5,
)


# ax.set_xscale("log")
# ax.set_yscale("log")

fig.savefig("./figs/fesc_vs_neutral_gas_density.png")


# fig,ax = make_figure()

# age_bins = np.linspace(0, 100, 50)

# codaiii_mean_lintr, bins, counts = binned_statistic(codaiii_datas["stellar age"], codaiii_datas["halo lintr"], statistic="mean", bins=age_bins)
# dustier_mean_lintr, bins, counts = binned_statistic(dustier_datas["age"], dustier_datas["lintr"], statistic="mean", bins=age_bins)

# ax.plot(bins[:-1]+np.diff(bins)*0.5, codaiii_mean_lintr, lw=3, label="CoDa III")
# ax.plot(bins[:-1]+np.diff(bins)*0.5, dustier_mean_lintr, lw=3, label="DUSTiER")

# # ax.set_xscale("log")
# ax.set_yscale("log")

# ax.set_ylabel(r"$L_{\mathrm{intr}}$")
# ax.set_xlabel(r"Age, Myr")
# ax.legend()

# fig.savefig("./figs/lintr_vs_age.png")

# print(dustier_keys)

fig, ax = make_figure()

mag_bins = np.linspace(-10, 15, 50)

# plot  median extincted and non extincted beta magnitude relations from codaiii and dustier
codaiii_mean_mag, bins, counts = binned_statistic(
    codaiii_datas["stellar mag no_dust"],
    codaiii_datas["stellar beta no_dust"],
    statistic="median",
    bins=mag_bins,
)
dustier_mean_mag, bins, counts = binned_statistic(
    dustier_datas["mag"], dustier_datas["beta"], statistic="median", bins=mag_bins
)

codaiii_mean_mag_ext, bins, counts = binned_statistic(
    codaiii_datas["stellar mag kext_albedo_WD_LMCavg_20"],
    codaiii_datas["stellar beta kext_albedo_WD_LMCavg_20"],
    statistic="median",
    bins=mag_bins,
)
dustier_mean_mag_ext, bins, counts = binned_statistic(
    dustier_datas["mag ext_LMC"],
    dustier_datas["beta ext_LMC"],
    statistic="median",
    bins=mag_bins,
)

ax.plot(
    bins[:-1] + np.diff(bins) * 0.5,
    codaiii_mean_mag,
    lw=3,
    label="CoDa III",
    c="tab:blue",
)
ax.plot(
    bins[:-1] + np.diff(bins) * 0.5,
    dustier_mean_mag,
    lw=3,
    label="DUSTiER",
    c="tab:orange",
)

ax.plot(
    bins[:-1] + np.diff(bins) * 0.5,
    codaiii_mean_mag_ext,
    lw=3,
    ls="--",
    label="CoDa III, extincted",
    c="tab:blue",
)
ax.plot(
    bins[:-1] + np.diff(bins) * 0.5,
    dustier_mean_mag_ext,
    lw=3,
    ls="--",
    label="DUSTiER, extincted",
    c="tab:orange",
)

# overlay 2D codaiii point density image in blue
codaiii_mag_beta_hist, xedges, yedges = np.histogram2d(
    codaiii_datas["stellar mag kext_albedo_WD_LMCavg_20"],
    codaiii_datas["stellar beta kext_albedo_WD_LMCavg_20"],
    bins=(mag_bins, np.linspace(-3, -1, 50)),
)
codaiii_mag_beta_hist = np.ma.masked_where(
    codaiii_mag_beta_hist == 0, codaiii_mag_beta_hist
)

ax.imshow(
    codaiii_mag_beta_hist.T,
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    origin="lower",
    aspect="auto",
    cmap="Blues",
    alpha=0.5,
)

# overlay 2D dustier point density image in orange
dustier_mag_beta_hist, xedges, yedges = np.histogram2d(
    dustier_datas["mag ext_LMC"],
    dustier_datas["beta ext_LMC"],
    bins=(mag_bins, np.linspace(-3, -1, 50)),
)
dustier_mag_beta_hist = np.ma.masked_where(
    dustier_mag_beta_hist == 0, dustier_mag_beta_hist
)

ax.imshow(
    dustier_mag_beta_hist.T,
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    origin="lower",
    aspect="auto",
    cmap="Oranges",
    alpha=0.5,
)


ax.set_ylabel(r"$\beta$")
ax.set_xlabel(r"$M_\mathrm{UV}$")

ax.set_xlim(-10, -3)
ax.set_ylim(-3, -1)

ax.legend()

fig.savefig("./figs/beta_vs_mag.png")


# plot extinction as a function of dust density in host cell

fig, ax = make_figure()

dust_bins = np.logspace(-30, -25, 25)

# coda_fesc_filter = codaiii_datas["stellar fesc no_dust"]<1.0 #files not done yet

print(codaiii_datas["stellar fesc no_dust"])

codaiii_mean_fesc, bins, counts = binned_statistic(
    codaiii_datas["dust density"],
    codaiii_datas["stellar mag kext_albedo_WD_LMCavg_20"]
    - codaiii_datas["stellar mag no_dust"],
    statistic="mean",
    bins=dust_bins,
)
dustier_mean_fesc, bins, counts = binned_statistic(
    dustier_datas["dust"],
    dustier_datas["mag ext_LMC"] - dustier_datas["mag"],
    statistic="mean",
    bins=dust_bins,
)

ax.plot(bins[:-1] + np.diff(bins) * 0.5, codaiii_mean_fesc, lw=3, label="CoDa III")
ax.plot(bins[:-1] + np.diff(bins) * 0.5, dustier_mean_fesc, lw=3, label="DUSTiER")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_ylabel(r"Extinction")
ax.set_xlabel(r"$\rho_\mathrm{dust}$, density code units")
ax.legend()

# ax.set_ylim(1e-4, 1.0)

fig.savefig("./figs/ext_vs_dust_density.png")


# plot reddening as a function of dust density in host cell

fig, ax = make_figure()

dust_bins = np.logspace(-30, -25, 25)

# coda_fesc_filter = codaiii_datas["stellar fesc no_dust"]<1.0 #files not done yet

# print(codaiii_datas["stellar fesc no_dust"])

codaiii_mean_fesc, bins, counts = binned_statistic(
    codaiii_datas["dust density"],
    codaiii_datas["stellar beta kext_albedo_WD_LMCavg_20"]
    - codaiii_datas["stellar beta no_dust"],
    statistic="mean",
    bins=dust_bins,
)
dustier_mean_fesc, bins, counts = binned_statistic(
    dustier_datas["dust"],
    dustier_datas["beta ext_LMC"] - dustier_datas["beta"],
    statistic="mean",
    bins=dust_bins,
)


ax.plot(bins[:-1] + np.diff(bins) * 0.5, codaiii_mean_fesc, lw=3, label="CoDa III")
ax.plot(bins[:-1] + np.diff(bins) * 0.5, dustier_mean_fesc, lw=3, label="DUSTiER")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_ylabel(r"UV slope reddening")
ax.set_xlabel(r"$\rho_\mathrm{dust}$, density code units")
ax.legend()

ax.set_ylim(1e-4, 1.0)

fig.savefig("./figs/reddening_vs_dust_density.png")


# plot fesc no dust as a function of stellar radii

fig, ax = make_figure()
radii_bins = np.logspace(0, 3, 500)

codaiii_mean_fesc, bins, counts = binned_statistic(
    codaiii_datas["stellar radii"],
    codaiii_datas["stellar fesc no_dust"],
    statistic="mean",
    bins=radii_bins,
)
dustier_mean_fesc, bins, counts = binned_statistic(
    dustier_datas["radius"], dustier_datas["fesc"], statistic="mean", bins=radii_bins
)

ax.plot(bins[:-1] + np.diff(bins) * 0.5, codaiii_mean_fesc, lw=3, label="CoDa III")
ax.plot(bins[:-1] + np.diff(bins) * 0.5, dustier_mean_fesc, lw=3, label="DUSTiER")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_ylabel(r"fesc")
ax.set_xlabel(r"radii in cells")
ax.legend()


fig.savefig("./figs/fesc_v_radii.png")


# plot mean reltionship between gas density and radii

fig, ax = make_figure()

gas_bins = np.logspace(-30, -24, 60)

# add 1d histograms of gas density and radii
divider = make_axes_locatable(ax)
yax = divider.append_axes("right", size="30%", pad=0.05, sharey=ax)
xax = divider.append_axes("top", size="30%", pad=0.05, sharex=ax)

xax.hist(
    codaiii_datas["gas density"],
    bins=gas_bins,
    histtype="step",
    color="tab:blue",
    orientation="vertical",
    density=True,
)

yax.hist(
    codaiii_datas["stellar radii"],
    bins=radii_bins,
    histtype="step",
    color="tab:blue",
    orientation="horizontal",
    density=True,
)

yax.hist(
    dustier_datas["radius"],
    bins=radii_bins,
    histtype="step",
    color="tab:orange",
    orientation="horizontal",
    density=True,
)

xax.hist(
    dustier_datas["gas density"],
    bins=gas_bins,
    histtype="step",
    color="tab:orange",
    orientation="vertical",
    density=True,
)

xax.tick_params(labelbottom=False)
yax.tick_params(labelleft=False)

yax.set_xscale("log")
xax.set_yscale("log")

codaiii_mean_fesc, bins, counts = binned_statistic(
    codaiii_datas["gas density"],
    codaiii_datas["stellar radii"],
    statistic="mean",
    bins=gas_bins,
)
dustier_mean_fesc, bins, counts = binned_statistic(
    dustier_datas["gas density"],
    dustier_datas["radius"],
    statistic="mean",
    bins=gas_bins,
)

ax.plot(bins[:-1] + np.diff(bins) * 0.5, codaiii_mean_fesc, lw=3, label="CoDa III")
ax.plot(bins[:-1] + np.diff(bins) * 0.5, dustier_mean_fesc, lw=3, label="DUSTiER")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_ylabel(r"radius")
ax.set_xlabel(r"gas density, $\mathrm{cm}^{-3}$")
ax.legend()

ax.set_ylim(1, 60)


fig.savefig("./figs/radii_v_rho.png")

# plot mean reltionship between dust density and radii

fig, ax = make_figure()

dust_bins = np.logspace(-30, -24, 60)


# add 1d histograms of dust density and radii
divider = make_axes_locatable(ax)
yax = divider.append_axes("right", size="30%", pad=0.05, sharey=ax)
xax = divider.append_axes("top", size="30%", pad=0.05, sharex=ax)

xax.hist(
    codaiii_datas["dust density"],
    bins=dust_bins,
    histtype="step",
    color="tab:blue",
    orientation="vertical",
    density=True,
)

yax.hist(
    codaiii_datas["stellar radii"],
    bins=radii_bins,
    histtype="step",
    color="tab:blue",
    orientation="horizontal",
    density=True,
)

yax.hist(
    dustier_datas["radius"],
    bins=radii_bins,
    histtype="step",
    color="tab:orange",
    orientation="horizontal",
    density=True,
)

xax.hist(
    dustier_datas["dust"],
    bins=dust_bins,
    histtype="step",
    color="tab:orange",
    orientation="vertical",
    density=True,
)


codaiii_mean_fesc, bins, counts = binned_statistic(
    codaiii_datas["dust density"],
    codaiii_datas["stellar radii"],
    statistic="mean",
    bins=gas_bins,
)
dustier_mean_fesc, bins, counts = binned_statistic(
    dustier_datas["dust"],
    dustier_datas["radius"],
    statistic="mean",
    bins=gas_bins,
)
xax.set_yscale("log")
yax.set_xscale("log")

xax.tick_params(labelbottom=False)
yax.tick_params(labelleft=False)

ax.plot(bins[:-1] + np.diff(bins) * 0.5, codaiii_mean_fesc, lw=3, label="CoDa III")
ax.plot(bins[:-1] + np.diff(bins) * 0.5, dustier_mean_fesc, lw=3, label="DUSTiER")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_ylabel(r"radius")
ax.set_xlabel(r"gas density, $\mathrm{cm}^{-3}$")
ax.legend()

ax.set_ylim(1, 60)

fig.savefig("./figs/radii_v_rho_dust.png")
