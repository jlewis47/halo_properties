import numpy as np
import matplotlib.pyplot as plt

# from scipy.stats import binned_statistic
from halo_properties.utils.utils import gather_h5py_files
from halo_properties.utils.output_paths import gen_paths, dataset
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos

# from plot_functions.generic.stat import xy_stat
from plot_functions.generic.plot_functions import make_figure, setup_plotting
from plot_functions.ionising.lescs import plot_ndot, plot_cosmic_ndot_constraints
import os
import h5py
from scipy.spatial import KDTree


def load_data(
    sim_name,
    out_nb,
    dset,
    ifile=None,
    fesc_type="gas",
    xkey="mass",
):
    fesc_keys = {"gas": "Tr_no_dust", "full": "Tr_kext_albedo_WD_LMC2_10"}

    fesc_key = fesc_keys[fesc_type]

    # print(f'LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')

    keys = [xkey, fesc_key, "Lintr"]

    out, assoc_out, analy_out, suffix = gen_paths(sim_name, out_nb, dset)

    # print(analy_out)

    # try:
    if ifile == None:
        datas = gather_h5py_files(analy_out, keys=keys)
    else:
        datas = {}

        fname = f"halo_stats_{ifile:d}"

        tgt_path = os.path.join(analy_out, fname)

        # print(fname)
        with h5py.File(tgt_path, "r") as src:
            for k in keys:
                try:
                    datas[k] = src[k][()]
                except KeyError:
                    print(f"no key file {fname:s}")
                    return [np.nan]
    # except OSError as e:
    # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association.: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
    # print(e)

    return (datas[fesc_key], datas["Lintr"])


setup_plotting()

# out_nbs = [14, 23, 34, 42, 52, 65, 82, 106]
# out_nbs = [34, 42, 52, 65, 82, 106]
out_nbs = [106]

overwrite = False
fesc_type = "gas"
x_type = "mass"
stat_mthd = "mean"
ll = 0.2
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthd = "stellar_peak"
# assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
r200 = 1.0
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]
mp = False
clean = True

l = 64

fperside = int(ldx / 512)
Nsub_pside = int(64 / fperside)
Nsub = int(Nsub_pside**3)
fpersub = int(4096 / Nsub)
fpersub_pside = int(np.cbrt(fpersub))
subLco = l / Nsub_pside

print(f"fpersub : {fpersub:d}, Nsub : {Nsub:d} Nsub_pside : {Nsub_pside:d}")

lesc_global_avg = np.zeros_like(out_nbs, dtype="f8")
lesc_global_std = np.zeros_like(out_nbs, dtype="f8")
lesc_global_p50 = np.zeros_like(out_nbs, dtype="f8")
lesc_global_p5 = np.zeros_like(out_nbs, dtype="f8")
lesc_global_p95 = np.zeros_like(out_nbs, dtype="f8")
redshifts = np.zeros_like(out_nbs, dtype="f4")

labels = []
lines = []

fig, ax = make_figure()


out_path = os.path.join("./files")
if not os.path.isdir(out_path):
    os.makedirs(out_path)

out_file = os.path.join(
    out_path,
    f"cosmic_lesc_{stat_mthd:s}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{fesc_type:s}",
)

if mp:
    out_file += "_mp"
if clean:
    out_file += "_clean"

exists = os.path.isfile(out_file)


dset = dataset(
    rtwo=r200,
    ll=ll,
    assoc_mthd=assoc_mthd,
    clean=clean,
    mp=mp,
    neb_cont_file_name=neb_cont_file_name,
)

if not exists or overwrite:
    cosmic_lesc = np.zeros((len(out_nbs), Nsub))
    redshifts = np.zeros(len(out_nbs))

    for i_out, out in enumerate(out_nbs):
        print(f"outnb : {out:d}")

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
        ) = get_infos(
            os.path.join(sim_path, f"output_{out:06d}/group_000001"), out, ldx
        )
        redshifts[i_out] = 1.0 / a - 1.0

        for isub in range(Nsub):
            ix, iy, iz = np.unravel_index(isub, (Nsub_pside, Nsub_pside, Nsub_pside))
            ixp1, iyp1, izp1 = ix + 1, iy + 1, iz + 1

            # print(ix,ixp1)
            # print(iy,iyp1)
            # print(iz,izp1)

            xs = np.arange(ix * fpersub_pside, ixp1 * fpersub_pside)
            ys = np.arange(iy * fpersub_pside, iyp1 * fpersub_pside)
            zs = np.arange(iz * fpersub_pside, izp1 * fpersub_pside)

            # print(xs, ys, zs)

            ifiles = np.ravel(
                np.ravel_multi_index(
                    np.meshgrid(xs, ys, zs), (fperside, fperside, fperside)
                )
            )

            # print(ifiles)

            for ifile in ifiles:
                fesc, lintr = load_data(
                    sim_name,
                    out,
                    dset,
                    ifile=ifile,
                )
                cosmic_lesc[i_out, isub] += np.nansum(fesc * lintr) / (
                    subLco**3 / (H0 / 100) ** 3
                )  # Msun per comoving Mpc
            # print((ix+1)/2.,(ix)/2.)

    lesc_global_avg = np.mean(cosmic_lesc, axis=1)
    # lesc_global_med=np.median(cosmic_lesc,axis=1)
    lesc_global_std = np.std(cosmic_lesc, axis=1)
    # lesc_global_iqr=stats.iqr(cosmic_lesc,axis=1)

    lesc_global_p5 = np.percentile(cosmic_lesc, 5, axis=1)
    lesc_global_p50 = np.percentile(cosmic_lesc, 50, axis=1)
    lesc_global_p95 = np.percentile(cosmic_lesc, 95, axis=1)

    with h5py.File(out_file, "w") as dest:
        dest.create_dataset("redshifts", data=redshifts)
        dest.create_dataset("avg", data=lesc_global_avg)
        dest.create_dataset("med", data=lesc_global_p50)
        dest.create_dataset("std", data=lesc_global_std)
        dest.create_dataset("lo_5", data=lesc_global_p5)
        dest.create_dataset("hi_95", data=lesc_global_p95)

else:
    with h5py.File(out_file, "r") as src:
        redshifts = src["redshifts"][()]
        lesc_global_avg = src["avg"][()]
        lesc_global_std = src["std"][()]
        lesc_global_p50 = src["med"][()]
        lesc_global_p5 = src["lo_5"][()]
        lesc_global_p95 = src["hi_95"][()]

    # labels.append(f"{stat_mthd:s} {assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200")


print(redshifts, lesc_global_avg)

line = plot_ndot(
    fig,
    ax,
    redshifts,
    lesc_global_avg,
    lo=lesc_global_p5,
    hi=lesc_global_p95,
)
cst_lines, cst_labels = plot_cosmic_ndot_constraints(ax, redshifts)
# dustier_lines, dustier_labels = plot_(ax, redshift, fkey="fesc")

# labels += dustier_labels
# lines += dustier_lines
# print(cst_lines, cst_labels)

ax.grid()
ax.legend(
    [line] + cst_lines,
    ["CoDa III"] + cst_labels,
    framealpha=0.1,
    prop={"size": 7},
)
# ax.legend([line], ["CoDa III"], framealpha=0.1)

ax.grid()

ax.set_xlim(15, 4.6)
ax.set_ylim(1e50, 2e51)

fig_name = f"./figs/cosmic_lesc_{stat_mthd:s}_{assoc_mthd:s}_ll={int(100*ll):d}_{int(r200):d}Xr200_{fesc_type:s}"
if mp:
    fig_name += "_mp"
if clean:
    fig_name += "_clean"

fig.savefig(fig_name)
fig.savefig(fig_name + ".pdf", format="pdf")
