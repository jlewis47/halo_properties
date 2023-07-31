import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from halo_properties.utils.utils import gather_h5py_files, ll_to_fof_suffix, get_r200_suffix, get_suffix
from halo_properties.utils.output_paths import gen_paths
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.generic.stat import xy_stat
from plot_functions.generic.plot_functions import make_figure
from plot_functions.UV.extinction import make_ext, plot_ext, plot_dustier_ext
import os
import h5py



def load_mags(sim_name, out_nb, assoc_mthd, ll, rtwo_fact, ext_key):


    print(f'LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {ext_key:s}')       

    

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix, rtwo_suffix)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=[ext_key])
    # except OSError as e:
        # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
        # print(e)
        

    return(datas[ext_key][()])

# out_nbs=[14,23,34,42,52,65,82]
out_nbs=[14,23,34,42,52,65,82,106]
overwrite = False 
stat_mthd = 'median'
lls = [0.2]#[0.1]
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthds = ['stellar_peak']
# assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
r200s = [1.0]
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]

ext_keys = get_dust_att_keys()
mag_ext_keys = ['mag_'+k for k in ext_keys]


nbins = 40
mag_bins = np.linspace(-25, -5, nbins)

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


    redshift = 1./a - 1.

    assert len(r200s) == len(assoc_mthds) == len(lls), "check input parameter lists' lengths"

    masses = []
    fescs = []
    labels = []

    fig, ax = make_figure()

    ncols=6
    nrows=1


    for iplot, (assoc_mthd, ll, r200) in enumerate(zip(assoc_mthds, lls, r200s)):
        
        out_path = os.path.join("./files")
        if not os.path.isdir(out_path):os.makedirs(out_path)

        out_file = os.path.join(out_path, f'fescs_{redshift:.1f}_{stat_mthd:s}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}')

        exists = os.path.isfile(out_file)

        if overwrite or not exists:

            mass, fesc = load_mags("CoDaIII", out_nb, assoc_mthd, ll, r200, ext_key=ext_key)
            xbins, ystat = xy_stat(mass, fesc, xbins = mag_bins, mthd=stat_mthd)
        

            # if not exists :
            with h5py.File(out_file, 'w') as dest:
                dest.create_dataset("xbins", data = xbins, dtype='f4')
                dest.create_dataset("fescs", data = ystat, dtype='f8')

            # elif overwrite and exists:

            #     with h5py.File(out_file, 'a') as dest:
            #         f_masses = dest["xbins"]
            #         f_fescs = dest["fescs"]

            #         f_masses[...] = xbins
            #         f_fescs[...] = ystat

        else:

            with h5py.File(out_file, 'r') as dest:
                xbins = dest["xbins"][()]
                ystat = dest["fescs"][()]

        masses.append(xbins)
        fescs.append(ystat)

        labels.append(f"{stat_mthd:s} {assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200")

        lines = plot_ext(fig, ax, masses, fescs, fesc_type, redshift)

        # dustier_lines, dustier_labels = plot_dustier_fesc(ax, redshift, fkey="fesc")

        # labels += dustier_labels
        # lines += dustier_lines


plt.legend(lines, labels, framealpha=0.0)

fig_name = f'./figs/fesc_comparison_{out_nb:d}'
fig.savefig(fig_name)




