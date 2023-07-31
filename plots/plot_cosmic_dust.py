from plot_functions.dust.cosmic_dust import plot_cosmic_dust, plot_dustier_cosmic_dust, plot_constraints
from plot_functions.generic.plot_functions import make_figure, setup_plotting

# from halo_properties.files.read_avgs import read_avgs
from halo_properties.params.params import *
# from halo_properties.utils.h5py_fcts import gather_h5py_files
from halo_properties.utils.utils import gather_h5py_files, ll_to_fof_suffix, get_r200_suffix, get_suffix
from halo_properties.utils.output_paths import gen_paths
from halo_properties.utils.functions_latest import get_infos

import os
import numpy as np
import h5py


# avgs = read_avgs(sim_path, "averages00000.txt")

# dust_avg = avgs['d'] #g/cm^3
# dust_avg = dust_avg * (pc * 1e3 * 1e2)**3 * 1e3 / Msol

# print(dust_avg, 1./avgs['aexp']-1)

def load_dust(sim_name, out_nb, assoc_mthd, ll, rtwo_fact, ifile):



    # print(f'LOADING : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}')       

    keys = ['Md']

    fof_suffix = ll_to_fof_suffix(ll)
    rtwo_suffix = get_r200_suffix(rtwo_fact)
    suffix = get_suffix(fof_suffix, rtwo_suffix)

    out, assoc_out, analy_out = gen_paths(sim_name, out_nb, suffix, assoc_mthd)

    fname = f'halo_stats_{ifile:d}'

    with h5py.File(os.path.join(analy_out, fname), 'r') as src:
        try:
            Mds = src['Md'][()]
        except KeyError:
            print(f'no key file {fname:s}')
            return([np.nan])

    return(Mds)


out_nbs=[14,23,34,42,52,65,82,106]

Nsub=int(4096/8.)
subLco = 64**3/Nsub
overwrite=True


tgt_file = './files/codaiii_dust_densities.hdf5'

if not os.path.exists(tgt_file) or overwrite:

    cosmic_dust = np.zeros((len(out_nbs), Nsub))
    redshifts = np.zeros(len(out_nbs))

    for i_out,out in enumerate(out_nbs):
    
        print(f"outnb : {out:d}")


        t,a,H0,om_m,om_l,om_k,om_b,unit_l,unit_d,unit_t,l,Lco,L,px_to_m = get_infos(os.path.join(sim_path,f"output_{out:06d}/group_000001"), out, ldx)
        redshifts[i_out] = 1./a - 1.

        for isub in range(Nsub):

            # ix,iy,iz = np.unravel_index(isub, (8,8,8))

            for ifile in range(isub*8,(isub+1)*8):
                Mds = load_dust(sim_name, out, assoc_mthd='stellar_peak', ll=0.2, rtwo_fact=1.0, ifile=ifile)
                cosmic_dust[i_out,isub]+=np.nansum(Mds)/(subLco) #Msun per comoving Mpc.h^-1
            #print((ix+1)/2.,(ix)/2.)
            






    dust_global_avg=np.mean(cosmic_dust,axis=1)
    #dust_global_med=np.median(cosmic_dust,axis=1)
    dust_global_std=np.std(cosmic_dust,axis=1)
    #dust_global_iqr=stats.iqr(cosmic_dust,axis=1)


    dust_global_16=np.percentile(cosmic_dust, 16, axis=1)
    dust_global_84=np.percentile(cosmic_dust, 84, axis=1)

    with h5py.File(tgt_file, 'w') as dest:
        dest.create_dataset('redshifts', data=redshifts)
        dest.create_dataset('avg', data=dust_global_avg)
        dest.create_dataset('std', data=dust_global_std)
        dest.create_dataset('lo_16', data=dust_global_16)
        dest.create_dataset('hi_84', data=dust_global_84)

else:

    with h5py.File(tgt_file, 'r') as src:
        redshifts=src['redshifts'][()]
        dust_global_avg=src['avg'][()]
        dust_global_std=src['std'][()]
        dust_global_16=src['lo_16'][()]
        dust_global_84=src['hi_84'][()]

setup_plotting()

fig, ax= make_figure()

plot_cosmic_dust(fig, ax, redshifts, dust_global_avg, dust_global_16, dust_global_84, dust_global_std)

plot_dustier_cosmic_dust(fig, ax)

plot_constraints(fig, ax)

ax.set_xlim(12,0)
ax.set_ylim(1e-1,2e6)
ax.autoscale_view(scalex=False,scaley=True)

fig.savefig('figs/cosmic_dust')
fig.savefig('figs/cosmic_dust.pdf',format='pdf')