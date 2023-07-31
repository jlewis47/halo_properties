import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from halo_properties.utils.utils import gather_h5py_files, ll_to_fof_suffix, get_r200_suffix, get_suffix
from halo_properties.utils.output_paths import gen_paths
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.dust.msmd import plot_msmd_constraints, plot_msmd, plot_dustier_msmd
from plot_functions.generic.stat import xy_stat
import os
import h5py



def load_data(sim_name, out_nb, assoc_mthd, ll, rtwo_fact):


    print(f'LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}')       

    

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix, rtwo_suffix)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    keys=['Mst', 'Md']

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
        # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
        # print(e)
        

    return([datas[key][()] for key in keys])

out_nbs=[34,42,52,65,82,106]
overwrite = False 
stat_mthd = 'median'
lls = [0.2]#[0.1]
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthds = ['stellar_peak']
# assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
r200s = [1.0]
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]


nbins = 55
mbins = np.logspace(3.75, 10.75, nbins)


ncols=6
nrows=1
nplots=len(out_nbs)
while ncols>4:

    ncols=int(np.ceil(nplots/float(nrows)))

    nrows+=1

nrows-=1

fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(25,12))
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


    redshift = 1./a - 1.

    assert len(r200s) == len(assoc_mthds) == len(lls), "check input parameter lists' lengths"
    
    ax = axs[iplot]

    for assoc_mthd, ll, r200 in zip(assoc_mthds, lls, r200s):
        
        

        out_path = os.path.join("./files")
        if not os.path.isdir(out_path):os.makedirs(out_path)

        out_file = os.path.join(out_path, f'MdMs_{out_nb:d}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}')


        exists = os.path.isfile(out_file)

        if overwrite or not exists:
        
            Mst, Md = load_data("CoDaIII", out_nb, assoc_mthd, ll, r200)
        
            # print(mags.min(), mags.max(), np.mean(mags))
        
            bins, med = xy_stat(Mst, Md, mbins, mthd=stat_mthd)

            with h5py.File(out_file, 'w') as dest:
                dest.create_dataset("stellar mass", data = bins, dtype='f4')
                dest.create_dataset("dust mass", data = med, dtype='f8')
                


        else:

            with h5py.File(out_file, 'r') as dest:
                bins = dest["stellar mass"][()]
                med = dest["dust mass"][()]
                


        # print(list(zip(bins, uvlf, err)))
        # print()
        line = plot_msmd(fig, ax, bins, med, linewidth=3, elinewidth=2, capsize=5)
        label = f"{stat_mthd:s} {assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200"

        lines.append([line])
        labels.append([label])


    obs_lines, obs_labels = plot_msmd_constraints(ax, redshift)
    dustier_lines ,dustier_labels = plot_dustier_msmd(ax, redshift, color='k')

    ax.legend(lines+obs_lines+dustier_lines, labels+obs_labels+dustier_labels, framealpha=0.0, title=f"z={redshift:.1f}")


# ax.set_ylim(1, 1e7)
# ax.set_xlim(-5,-23)

# if nplots%2==0:
#     ax.invert_xaxis()


for idel in range(iplot+1, nrows*ncols):

    axs[idel].remove()



fig.suptitle(f"{assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200")


fig_name = f'./figs/MstMd_comparison_panel'
fig.savefig(fig_name)





