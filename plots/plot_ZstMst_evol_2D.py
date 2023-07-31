import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from halo_properties.utils.utils import gather_h5py_files, ll_to_fof_suffix, get_r200_suffix, get_suffix
from halo_properties.utils.output_paths import gen_paths
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.generic.stat import xy_2Dstat, xy_stat
from plot_functions.generic.plot_functions import density_plot_fancy
from plot_functions.dust.zstmst import plot_zstmst_constraints, plot_zstmst, plot_dustier_zstmst
import os
import h5py



def load_data(sim_name, out_nb, assoc_mthd, ll, rtwo_fact):


    print(f'LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}')       

    

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix, rtwo_suffix)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    keys=['Mst', 'stZ_wMst']

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
        # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
        # print(e)
        

    return([datas[key][()] for key in keys])

out_nbs = [14,23,34,42,52,65,82,106]
overwrite = True 
# stat_mthd = 'count'
ll = 0.2
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthd = 'stellar_peak'
# assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
r200 = 1.0
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]

nmbins = 40
mass_bins = np.logspace(4, 10.5, nmbins)
nzbins = 40
zbins = np.logspace(-4, 0, nzbins)



ncols=6
nrows=1
nplots=len(out_nbs)
while ncols>4:

    ncols=int(np.ceil(nplots/float(nrows)))

    nrows+=1

nrows-=1

fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(29.7,21.0))
axs = np.ravel(axs)

for iplot, out_nb in enumerate(out_nbs):

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


    redshift = 1./a - 1.
        
    
    ax = axs[iplot]

    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):os.makedirs(out_path)

    out_file = os.path.join(out_path, f'MstZst_hist_{redshift:.1f}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}')
    read_mode = 'a'
    exists = os.path.isfile(out_file)

    if overwrite or not exists:

        Mst, Zst = load_data("CoDaIII", out_nb, assoc_mthd, ll, r200)
        Zst /= 0.02

        
        xbins, ybins, stat_2D = xy_2Dstat(Mst, Zst, xbins = mass_bins, ybins = zbins, mthd="count")
        xbins, stat_1D = xy_stat(Mst, Zst, xbins=mass_bins, mthd="median")


        # if not exists :
        with h5py.File(out_file, 'w') as dest:
            dest.create_dataset("xbins", data = xbins, dtype='f4')
            dest.create_dataset("ybins", data = ybins, dtype='f4')
            dest.create_dataset("counts", data = stat_2D, dtype='f8')
            dest.create_dataset("stat_1D", data = stat_1D, dtype='f8')

        # elif overwrite and exists:

        #     with h5py.File(out_file, 'a') as dest:
        #         f_masses = dest["xbins"]
        #         f_fescs = dest["fescs"]

        #         f_masses[...] = xbins
        #         f_fescs[...] = ystat

    else:

        with h5py.File(out_file, 'r') as dest:
            xbins = dest["xbins"][()]
            ybins = dest["ybins"][()]
            stat_2D = dest["counts"][()]
            stat_1D = dest["stat_1D"][()]





    density_plot_fancy(fig, ax, stat_2D, stat_2D, xbins, ybins, xlabel="Stellar Mass", ylabel="Average halo solar Z", vmin=0,vmax=1e5)
    lines = plot_zstmst(fig, ax, np.log10(xbins), np.log10(stat_1D), color='r', xscale="linear", yscale='linear')
    # labels = "Median"

    dustier_lines, dustier_labels = plot_dustier_zstmst(ax, redshift, log=True, c='k')

    
    # ax.legend([lines]+dustier_lines, [labels]+dustier_labels, framealpha=0.0, title=f"z={redshift:.1f}",loc='lower right')
    # plot_Mh_fesc_constraints(ax, redshift)


for idel in range(iplot+1, nrows*ncols):

    axs[idel].remove()

# plt.suptitle(f"z = {redshift:.1f} - star forming only")

fig_name = f'./figs/MstZst_hist2D_evol'
fig.savefig(fig_name)




