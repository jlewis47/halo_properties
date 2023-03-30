import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from halo_properties.utils.utils import gather_h5py_files, ll_to_fof_suffix, get_r200_suffix, get_suffix
from halo_properties.utils.output_paths import gen_paths
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
import os
import h5py
from core.stat import xy_stat
from core.plot_functions import plot_xy




def fesc_plot(masses, fescs, labels, fesc_type, out_nb):


    fesc_labels = {'gas':"$\mathrm{f_{esc, g}}$",
                'dust':"$\mathrm{f_{esc, d}}$",
                'full':"$\mathrm{f_{esc, gxd}}$"}

    # plot_args = {'ls':'-', 'lw':3}

    fig, ax = xy_plot(masses, fescs, legend=True, labels=labels,
    xlabel='$\mathrm{Halo \, \, masses, \, M_\odot}$', ylabel=fesc_labels[fesc_type],
    xscale='log', yscale='log')        

    if not os.path.isdir('$HOME/plot_halo_codaiii/figs'): os.makedirs('$HOME/plot_halo_codaiii/figs')

    ax.set_title(f'snapshot {out_nb:d}, only star forming haloes')

    fig.savefig(f'$HOME/plot_halo_codaiii/figs/fesc_comparison_{fesc_type:s}_{out_nb:d}')

    return(fig, ax)


def load_fescs(sim_name, out_nb, assoc_mthd, ll, rtwo_fact, fesc_type='gas', xkey='mass'):

    fesc_keys = {'gas':"Tr_no_dust",
                'full':"Tr_kext_albedo_WD_LMC2_10"}

    fesc_key = fesc_keys[fesc_type]

    print(f'LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, associatfiles.: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')       

    keys = [xkey, fesc_key, "SFR10"]

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix, rtwo_suffix)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
        # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, associatfiles.: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
        # print(e)
        
    sfing = datas["SFR10"]>0

    return(datas[xkey][sfing], datas[fesc_keys[fesc_type]][sfing])

out_nb = 52
overwrite = False 
fesc_type = 'gas'
x_type = 'mass'
stat_mthd = 'median'
# lls = [0.1]
lls = [0.1, 0.15, 0.2, 0.2, 0.2]
# assoc_mthds = ['stellar_peak']
assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
# r200s = [1.0]
r200s = [1.0, 1.0, 1.0, 1.0, 2.0]

nbins = 55
mass_bins = np.logspace(7.5, 12, nbins)

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

masses = []
fescs = []
labels = []

for assoc_mthd, ll, r200 in zip(assoc_mthds, lls, r200s):
    
    

    mass, fesc = load_fescs("CoDaIII", out_nb, assoc_mthd, ll, r200, fesc_type=fesc_type, xkey='mass')

    out_path = os.path.join("../codaiii_halo_finding_compare/files")
    if not os.path.isdir(out_path):os.makedirs(out_path)

    out_file = os.path.join(out_path, f'fescs_{stat_mthd:s}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}')
    read_mode = 'a'
    exists = os.path.isfile(out_file)

    if overwrite or not exists:
        xbins, ystat = xy_stat(mass, fesc, xbins = mass_bins, mthd=stat_mthd)

        if not exists:
            with h5py.File(out_file, 'w') as dest:
                dest.create_dataset("xbins", data = xbins, dtype='f4')
                dest.create_dataset("fescs", data = fescs, dtype='f8')

        elif overwrite and exists:

            with h5py.File(out_file, 'a') as dest:
                f_masses = dest["xbins"]
                f_fescs = dest["fescs"]

                f_masses[...] = xbins
                f_fescs[...] = ystat

    else:

        with h5py.File(out_file, 'r') as dest:
            masses = dest["xbins"][()]
            fescs = dest["fescs"][()]



    masses.append(xbins)
    fescs.append(ystat)

 

    labels.append(f"{stat_mthd:s} {assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200")

    ax, fig = fesc_plot(masses, fescs, labels, fesc_type, out_nb)




