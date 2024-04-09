import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from halo_properties.utils.utils import gather_h5py_files
from halo_properties.utils.output_paths import gen_paths, dataset
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.generic.stat import xy_stat
from plot_functions.generic.plot_functions import make_figure, setup_plotting
from plot_functions.ionising.fescs import fesc_Mh_plot, plot_dustier_fesc, plot_Mh_fesc_constraints
import os
import h5py



def load_data(sim_name, out_nb, assoc_mthd, ll, rtwo_fact, fesc_type='gas', xkey='mass', mp=False, clean=False):

    fesc_keys = {'gas':"Tr_no_dust",
                'full':"Tr_kext_albedo_WD_LMC2_10"}

    fesc_key = fesc_keys[fesc_type]

    print(f'LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')       

    keys = [xkey, fesc_key, "SFR10"]

    dset = dataset(rtwo=rtwo_fact, ll=ll, assoc_mthd=assoc_mthd, clean=clean, mp=mp)

    out, assoc_out, analy_out, suffix = gen_paths(
        sim_name, out_nb, dset
    )

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
        # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, associatfiles.: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
        # print(e)
        
    sfing = datas["SFR10"]>0

    return(datas[xkey], datas[fesc_key], sfing)

setup_plotting()

out_nbs = [34,52,65,82]
# out_nbs = [106]

overwrite = False 
fesc_type = 'gas'
x_type = 'mass'
stat_mthd = 'mean'
ll = 0.2
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthd = 'stellar_peak'
# assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
r200 = 1.0
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]
mp=True
clean=True

mnbins = 55
mass_bins = np.logspace(7.5, 12, mnbins)




masses = []
fescs = []
fescs_sfing = []
labels = []
lines=[]
redshifts=[]

fig, ax = make_figure()

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
    redshifts.append(redshift)

    
    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):os.makedirs(out_path)

    out_file = os.path.join(out_path, f'fescs_{redshift:.1f}_{stat_mthd:s}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{fesc_type:s}')
    if mp:
        out_file += "_mp"
    if clean:
        out_file += "_clean"


    exists = os.path.isfile(out_file)

    if overwrite or not exists:

        data_mass, data_fesc, sfing = load_data("CoDaIII", out_nb, assoc_mthd, ll, r200, fesc_type=fesc_type, xkey='mass', mp=mp, clean=clean)
        xbins, fesc = xy_stat(data_mass, data_fesc, xbins = mass_bins, mthd=stat_mthd)
        xbins, fesc_sfing = xy_stat(data_mass[sfing], data_fesc[sfing], xbins = mass_bins, mthd=stat_mthd)
    

        # if not exists :
        with h5py.File(out_file, 'w') as dest:
            dest.create_dataset("xbins", data = xbins, dtype='f4', compression='lzf')
            dest.create_dataset("%s_fescs"%stat_mthd, data = fesc, dtype='f8', compression='lzf')
            dest.create_dataset("%s_fescs_sfing"%stat_mthd, data = fesc_sfing, dtype='f8', compression='lzf')

        # elif overwrite and exists:

        #     with h5py.File(out_file, 'a') as dest:
        #         f_masses = dest["xbins"]
        #         f_fescs = dest["fescs"]

        #         f_masses[...] = xbins
        #         f_fescs[...] = ystat

    else:

        with h5py.File(out_file, 'r') as dest:
            xbins = dest["xbins"][()]
            fesc = dest["%s_fescs"%stat_mthd][()]
            fesc_sfing = dest["%s_fescs_sfing"%stat_mthd][()]

    masses.append(xbins)
    fescs.append(fesc)
    fescs_sfing.append(fesc_sfing)

    # labels.append(f"{stat_mthd:s} {assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200")

for z, mass, fesc, fesc_sfing in zip(redshifts, masses, fescs, fescs_sfing):
    # print(mass, fesc, fesc_sfing)

    kern = [1./3, 1./3, 1./3]
    
    mass= np.convolve(mass, kern, mode="valid")
    fesc= np.convolve(fesc, kern, mode="valid")
    fesc_sfing= np.convolve(fesc_sfing, kern, mode="valid")

    lines.append(fesc_Mh_plot(fig, ax, mass, fesc, fesc_type, redshift=None))
    labels.append(f"{z:.1f}")
    fesc_Mh_plot(fig, ax, mass, fesc_sfing, fesc_type, redshift=None, ls='--', c=lines[-1].get_color())


lines_cst, labels_cst = plot_Mh_fesc_constraints(ax, 6.0)
lines += lines_cst
labels += labels_cst

# dustier_lines, dustier_labels = plot_dustier_fesc(ax, redshift, fkey="fesc")

# labels += dustier_labels
# lines += dustier_lines
plt.grid()
plt.legend(lines, labels, framealpha=0.0, prop={"size":14}, ncol=2)

fig_name = f"./figs/fesc_evolution_{stat_mthd:s}_{assoc_mthd:s}_ll={int(100*ll):d}_{int(r200):d}Xr200_{fesc_type:s}"
fig.savefig(fig_name)
fig.savefig(fig_name+".pdf", format='pdf')




