import numpy as np
import matplotlib.pyplot as plt
from halo_properties.utils.utils import gather_h5py_files, ll_to_fof_suffix, get_r200_suffix, get_suffix
from halo_properties.utils.output_paths import gen_paths
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.uvlf.uvlfs import uvlf_plot, make_uvlf, plot_constraints
from plot_functions.generic.plot_functions import make_figure

import os
import h5py



def load_mags(sim_name, out_nb, assoc_mthd, ll, rtwo_fact, ext):

    if ext:
        key="mag_kext_albedo_WD_LMCavg_20"
    else:
        key="mag_no_dust"

    print(f'LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}')       

    

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix, rtwo_suffix)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=[key])
    # except OSError as e:
        # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
        # print(e)
        

    return(datas[key][()])

out_nb = 65
overwrite = True 
ext = True
# lls = [0.1]
lls = [0.1, 0.15, 0.2, 0.2, 0.2]
# assoc_mthds = ['stellar_peak']
assoc_mthds = ['stellar_peak', 'stellar_peak', 'stellar_peak', 'fof_ctr', 'stellar_peak']
# r200s = [1.0]
r200s = [1.0, 1.0, 1.0, 1.0, 2.0]

fig, ax = make_figure()

nbins = 40
mag_bins = np.linspace(-25, -5, nbins)

# print(mag_bins)

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

xs = []
uvlfs = []
labels = []

for assoc_mthd, ll, r200 in zip(assoc_mthds, lls, r200s):
    
    



    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):os.makedirs(out_path)

    out_file = os.path.join(out_path, f'uvlfs_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}')
    if ext:out_file+='_ext'

    exists = os.path.isfile(out_file)

    if overwrite or not exists:
    
        mags = load_mags("CoDaIII", out_nb, assoc_mthd, ll, r200, ext)
    
        # print(mags.min(), mags.max(), np.mean(mags))
    
        bins, uvlf = make_uvlf(mags, mag_bins, Lco)

        with h5py.File(out_file, 'w') as dest:
            dest.create_dataset("xbins", data = bins, dtype='f4')
            dest.create_dataset("uvlf", data = uvlf, dtype='f8')


    else:

        with h5py.File(out_file, 'r') as dest:
            bins = dest["xbins"][()]
            uvlf = dest["uvlf"][()]



    xs.append(bins)
    uvlfs.append(uvlf)

    # print(bins, uvlf)

    labels.append(f"{assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200")

lines = uvlf_plot(fig, ax, xs, uvlfs, redshift)
obs_lines, obs_labels = plot_constraints(fig, ax, redshift)

labels += obs_labels
lines += obs_lines

ax.legend(lines, labels, framealpha=0.0)

fig_name = f'./figs/uvlf_comparison_{out_nb:d}'
if ext:fig_name+='_ext'

fig.savefig(fig_name)




