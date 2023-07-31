import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from halo_properties.utils.utils import gather_h5py_files, ll_to_fof_suffix, get_r200_suffix, get_suffix
from halo_properties.utils.output_paths import gen_paths
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.generic.stat import xy_2Dstat, xy_stat
from plot_functions.ionising.fescs import plot_fesc_density, fesc_Mh_plot, plot_Mh_fesc_constraints
import os
import h5py



def load_fescs(sim_name, out_nb, assoc_mthd, ll, rtwo_fact, fesc_type='gas', xkey='mass'):

    fesc_keys = {'gas':"Tr_no_dust",
                'full':"Tr_kext_albedo_WD_LMC2_10",
                'dust':["Tr_no_dust","Tr_kext_albedo_WD_LMC2_10"]}

    fesc_key = fesc_keys[fesc_type]

    print(f'LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_type:s}, xkey={xkey:s}')       

    if type(fesc_key)==list:
        keys = [xkey, *fesc_key, "SFR10"]
    else:
        keys = [xkey, fesc_key, "SFR10"]

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix, rtwo_suffix)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    # print(analy_out)

    # try:
    print(keys)

    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
        # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, associatfiles.: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
        # print(e)
        

    # print(np.sum(sfing), np.sum(sfing==False))
    sfing = datas["SFR10"]>0
    masses = datas[xkey][sfing]

    if type(fesc_key)!=list:

        fesc = datas[fesc_key][sfing]

    else:

        fesc_no_dust=datas[fesc_key[0]][sfing]
        fesc_dust=datas[fesc_key[1]][sfing]
        fesc = fesc_dust/fesc_no_dust

        print(fesc)


    return(masses, fesc)

out_nbs = [14,23,34,42,52,65,82,106]
overwrite = False 
fesc_type = 'gas'
x_type = 'mass'
# stat_mthd = 'count'
ll = 0.2
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthd = 'stellar_peak'
# assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
r200 = 1.0
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]

mnbins = 55
mass_bins = np.logspace(7.5, 12, mnbins)
fnbins = 45
fesc_bins = np.logspace(-4, 0, fnbins)



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

    out_file = os.path.join(out_path, f'fescs_{fesc_type:s}_mass_hist_{redshift:.1f}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}')
    read_mode = 'a'
    exists = os.path.isfile(out_file)

    if overwrite or not exists:

        mass, fesc = load_fescs("CoDaIII", out_nb, assoc_mthd, ll, r200, fesc_type=fesc_type, xkey='mass')

        fesc[fesc<fesc_bins.min()] = fesc_bins.min()
        
        xbins, ybins, stat_2D = xy_2Dstat(mass, fesc, xbins = mass_bins, ybins = fesc_bins, mthd="count")
        xbins, stat_1D = xy_stat(mass, fesc, xbins=mass_bins, mthd="median")


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





    plot_fesc_density(fig, ax, stat_2D, xbins, ybins, fesc_type, cb=False, vmin=0,vmax=1e5)
    lines = fesc_Mh_plot(fig, ax, np.log10(xbins), np.log10(stat_1D), fesc_type, log=False, color='r')
    labels = "Median"
    if fesc_type!="dust":
        obs_lines, obs_labels = plot_Mh_fesc_constraints(ax, redshift, log=True)
    else:
        obs_lines, obs_labels= [],[]
    #ax.text(10.7, np.log10(3e-4), f"{assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200", fontdict={"size":12}, va='center', ha='center', color='grey')
    ax.legend([lines]+obs_lines, [labels]+obs_labels, framealpha=0.0, title=f"z={redshift:.1f}",loc='lower right')
    # plot_Mh_fesc_constraints(ax, redshift)


for idel in range(iplot+1, nrows*ncols):

    axs[idel].remove()

# plt.suptitle(f"z = {redshift:.1f} - star forming only")

fig_name = f'./figs/fesc_{fesc_type:}_evolution_hist'
fig.savefig(fig_name)




