import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import binned_statistic
from halo_properties.utils.utils import gather_h5py_files, ll_to_fof_suffix, get_r200_suffix, get_suffix
from halo_properties.utils.output_paths import gen_paths
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.generic.stat import xy_stat
from plot_functions.generic.plot_functions import make_figure, setup_plotting
from plot_functions.ionising.budget import make_budget, plot_stack_budget
import os
import h5py



def load_data(sim_name, out_nb, assoc_mthd, ll, rtwo_fact, fesc_type='gas', xkey='mass'):

    fesc_keys = {'gas':"Tr_no_dust",
                'full':"Tr_kext_albedo_WD_LMC2_10"}

    fesc_key = fesc_keys[fesc_type]

    print(f'LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}')       


    keys = [fesc_key, 'Lintr']
    if type(xkey) == str:
        keys += [xkey]
    else:#list
        keys += xkey

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix, rtwo_suffix)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    # try:
    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
        # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, associatfiles.: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
        # print(e)
    
    return(datas[fesc_key] * datas['Lintr'], datas)

setup_plotting()

out_nbs = [14,23,34,42,52,65,82,106]
# out_nbs = [106]

overwrite = False 
fesc_type = 'gas'
# x_plot = 'mass'
leg_title = r'$M_{\rm AB1600}$'
x_plot = 'mag_kext_albedo_WD_LMCavg_20'
# x_types = ['mass','mag_kext_albedo_WD_LMCavg_20', 'stZ_wMst', 'stAge_wMst', 'SFR10']
stat_mthd = 'sum'
ll = 0.2
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthd = 'stellar_peak'
# assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
r200 = 1.0
# r200s = [1.0, 1.0, 1.0, 1.0, 2.0]


# plot_bins = np.logspace(7, 13, 6)
# plot_bins = np.linspace(-25, -5, 6)
# plot_bins = np.linspace(-25, -5, 6)
plot_bins = np.sort(np.asarray([0,-15,-17,-19,-20,-21,-25]))
# plot_bin_labels = ["%.1e-%.1e"%(mm1,mp1) for mm1,mp1 in zip(plot_bins[:-1], plot_bins[1:])]
plot_bin_labels = ["[%d,%d]"%(mm1,mp1) for mm1,mp1 in zip(plot_bins[:-1], plot_bins[1:])]
budget_evol = np.zeros((len(plot_bins)-1, len(out_nbs)))


labels = []
lines=[]
redshifts=[]

fig, ax = make_figure()

for i_out, out_nb in enumerate(out_nbs):

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

    out_file = os.path.join(out_path, f'budget_{redshift:.1f}_{stat_mthd:s}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{fesc_type:s}')

    exists = os.path.isfile(out_file)

    no_key=True

    if exists:
        no_key = x_plot not in h5py.File(out_file, 'r').keys()

    if overwrite or not exists or no_key:

        lesc, datas = load_data("CoDaIII", out_nb, assoc_mthd, ll, r200, fesc_type=fesc_type, xkey=x_plot)
        
        with h5py.File(out_file, 'a') as dest:


            xs = datas[x_plot]
            xmin, xmax = np.floor(xs.min()), np.ceil(xs.max())
            if np.any(xs<0):
                xbins = np.linspace(xmin, xmax, int(np.abs(xmax - xmin)))
            else:
                xbins = np.logspace(np.log10(xmin), np.log10(xmax), int(np.abs(np.log10(xmax) - np.log10(xmin))) * 4)

            # print(len(xs), len(lesc), xbins)
            xbins, lesc = xy_stat(xs, lesc, xbins = xbins, mthd=stat_mthd)    

            grp = dest.create_group(x_plot)
            grp.create_dataset("x", data = xbins, dtype='f4', compression='lzf')
            grp.create_dataset("lesc", data = lesc, dtype='f8', compression='lzf')


    else:

        with h5py.File(out_file, 'r') as dest:
            xbins = dest[x_plot]["x"][()]
            lesc = dest[x_plot]["lesc"][()]

    file_lesc = lesc

    #rebin to plot blins

    file_lesc, xbins, counts =  binned_statistic(xbins, file_lesc, 'sum', plot_bins)
    budget_evol[:,i_out] = file_lesc

    # labels.append(f"{stat_mthd:s} {assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200")

print(plot_bins, budget_evol, redshifts, plot_bin_labels)

plot_stack_budget(fig, ax, plot_bins, budget_evol, redshifts, plot_bin_labels, leg_title=leg_title, txt_size=8, cmap=matplotlib.cm.get_cmap('jet_r'))

# dustier_lines, dustier_labels = plot_dustier_lesc(ax, redshift, fkey="fesc")

# labels += dustier_labels
# lines += dustier_lines
# plt.grid()

ax.tick_params(which='both', top=False, bottom=False, left=False, right=False)

plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)

fig_name = f"./figs/budget_stack_{x_plot:s}_{assoc_mthd:s}_ll={int(100*ll):d}_{int(r200):d}Xr200_{fesc_type:s}"
fig.savefig(fig_name, bbox_inches='tight')
fig.savefig(fig_name+'.pdf',format='pdf', bbox_inches='tight')




