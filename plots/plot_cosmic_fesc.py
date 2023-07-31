import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from halo_properties.utils.utils import gather_h5py_files, ll_to_fof_suffix, get_r200_suffix, get_suffix
from halo_properties.utils.output_paths import gen_paths
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.generic.stat import xy_stat
from plot_functions.generic.plot_functions import make_figure, setup_plotting
from plot_functions.ionising.fescs import plot_cosmic_fesc, plot_cosmic_fesc_constraints
import os
import h5py



def load_data(sim_name, out_nb, assoc_mthd, ll, rtwo_fact, fesc_type='gas', xkey='mass'):

    fesc_keys = {'gas':"Tr_no_dust",
                'full':"Tr_kext_albedo_WD_LMC2_10"}

    fesc_key = fesc_keys[fesc_type]

    print(f'LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')       

    keys = [xkey, fesc_key, "SFR10"]

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix, rtwo_suffix)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=keys+["Lintr"])
    # except OSError as e:
        # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, associatfiles.: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
        # print(e)
        

    return(datas[fesc_key], datas['Lintr'])

setup_plotting()

out_nbs = [14,23,34,42,52,65,82,106]
# out_nbs = [106]

overwrite = True
fesc_type = 'gas'
x_type = 'mass'
stat_mthd = 'mean'
ll = 0.2
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthd = 'stellar_peak'
# assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
r200 = 1.0
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]





fescs = np.zeros_like(out_nbs, dtype='f8')
redshifts = np.zeros_like(out_nbs, dtype='f4')

labels = []
lines=[]

fig, ax = make_figure()


out_path = os.path.join("./files")
if not os.path.isdir(out_path):os.makedirs(out_path)

out_file = os.path.join(out_path, f'cosmic_fesc_{stat_mthd:s}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{fesc_type:s}')

exists = os.path.isfile(out_file)

if overwrite or not exists:
    for i_out,out_nb in enumerate(out_nbs):

        # if out_nb==52:
        #     overwrite=True
        # else:
        #     overwrite=False

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
        redshifts[i_out] = redshift

        


        fesc, lintr = load_data("CoDaIII", out_nb, assoc_mthd, ll, r200, fesc_type=fesc_type, xkey='mass')
            
        print(np.sum(fesc * lintr)/np.sum(lintr))
        # print(np.sum(lintr))
        

        fescs[i_out] = np.sum(fesc * lintr) / np.sum(lintr)

    # if not exists :
    with h5py.File(out_file, 'w') as dest:
        dest.create_dataset("redshifts", data = redshifts, dtype='f4', compression='lzf')
        dest.create_dataset("fescs", data = fescs, dtype='f8', compression='lzf')
        

            # elif overwrite and exists:

            #     with h5py.File(out_file, 'a') as dest:
            #         f_masses = dest["xbins"]
            #         f_fescs = dest["fescs"]

            #         f_masses[...] = xbins
            #         f_fescs[...] = ystat

else:

    with h5py.File(out_file, 'r') as dest:
        redshifts = dest["redshifts"][()]
        fescs = dest["fescs"][()]
        



    # labels.append(f"{stat_mthd:s} {assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200")


print(redshifts, fescs)

line = plot_cosmic_fesc(fig, ax, redshifts, fescs)
cst_lines, cst_labels = plot_cosmic_fesc_constraints(ax, redshifts)
# dustier_lines, dustier_labels = plot_dustier_lesc(ax, redshift, fkey="fesc")

# labels += dustier_labels
# lines += dustier_lines
# print(cst_lines, cst_labels)

ax.grid()
ax.legend([line]+cst_lines, ['CoDa III']+cst_labels, framealpha=0.1)

ax.set_xlim(15,5)

fig_name = f"./figs/cosmic_fesc_{stat_mthd:s}_{assoc_mthd:s}_ll={int(100*ll):d}_{int(r200):d}Xr200_{fesc_type:s}"
fig.savefig(fig_name)




