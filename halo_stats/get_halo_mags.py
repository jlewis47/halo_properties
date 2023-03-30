from association.read_assoc_latest import read_assoc, read_specific_stars
import numpy as np
from src.bpass_fcts import *
from utils.output_paths import gen_paths
import argparse
import os
from utils.functions_latest import get_infos, get_Mp
from params.params import *
from utils.utils import *
from mpi4py import MPI
import h5py

def get_halo_mags(out_nb,ldx,path,sim_name,rtwo_fact,ll=0.2,assoc_mthd='',overwrite=False):

        comm = MPI.COMM_WORLD
        Nproc = comm.Get_size()
        rank = comm.Get_rank()
        
        if rank == 0:
                print('eta_sn is %f'%eta_sn)
                print("Running on snapshot %s"%out_nb)
        
        check_assoc_keys(assoc_mthd)
        
        output_str='output_%06i'%out_nb

        star_path = os.path.join(path, "reduced/stars", output_str)
        
        info_path = os.path.join(path,output_str,'group_000001')

        fof_suffix=ll_to_fof_suffix(ll)
        rtwo_suffix=get_r200_suffix(rtwo_fact)
        suffix=get_suffix(fof_suffix,rtwo_suffix)
        
        Np_tot=ldx**3

        '''Get scale factor and co'''
        (t,a,H0,om_m,om_l,om_k,om_b,unit_l,unit_d,unit_t,l,Lco,L,px_to_m) = get_infos(info_path,
                                                                                           out_nb,
                                                                                      ldx)

        Mp = get_Mp(om_m,om_b,H0,Lco,int(ldx**3))

        out,assoc_out,analy_out=gen_paths(sim_name,out_nb,suffix,assoc_mthd)

        out_name = os.path.join(analy_out,'halo_mags'+suffix)
        if not os.path.exists(out_name) or overwrite :  

                halos, halo_star_ids, tot_star_nbs=read_assoc(out_nb,'CoDaIII',rtwo_fact=rtwo_fact,ll=ll,assoc_mthd=assoc_mthd)
                star_nbs = halos['nstar']
                
                #print(len(stars))
                
                nb_halos=len(halos)

                halo_stellar_mass = halos['mstar']
                halo_mags=np.zeros(nb_halos, dtype=np.float32)

                mags,xis,contbetalow,contbetahigh,beta,metal_bins,age_bins=get_mag_tab_BPASSV221_betas()

                mag_fct = get_mag_interp_fct(mags, age_bins, metal_bins)
                
                # print(np.shape(stars))

                #print(age_bins)
                
                # print(star_nbs, tot_star_nbs)

                # print(len(stars),tot_star_nbs[-1])
                
                # nmin, nmax, nperProc = divide_task(len(halos), Nproc, rank)
                # proc_halos = halos[nmin:nmax]
                # proc_halo_idx = np.arange(nmin,nmax)

                x,y,z,lsub = divide_task_space(ldx, Nproc, rank)

                #print(x,y,z,rank)
                #print(halos)
                #print(halos['x'][halos['x']>ldx],halos['x'][halos['x']<0])
                xcond = (halos["x"]>=x * lsub) * (halos["x"]<(x * lsub + lsub))
                ycond = (halos["y"]>=y * lsub) * (halos["y"]<(y * lsub + lsub))
                zcond = (halos["z"]>=z * lsub) * (halos["z"]<(z * lsub + lsub))
                sub_cond = xcond * ycond * zcond

                #print(np.argwhere(sub_cond))
                
                proc_halos = halos[sub_cond]
                proc_halo_idx = np.arange(0, nb_halos)[sub_cond]
                
                proc_no_stars = proc_halos['nstar'] > 0
                proc_halos = proc_halos[proc_no_stars]
                proc_halo_idx = proc_halo_idx[proc_no_stars]

                #print(proc_halo_idx)
                
                for ihalo, halo in zip(proc_halo_idx, proc_halos):

                        #print(ihalo, halo['mass'] * Mp, halo['mstar'])
                                
                        #if rank == 0 and ((ihalo - proc_halo_idx[0]) % 1000 == 0.0) : print("...%.3f"%((ihalo - proc_halo_idx[0]) / (len(proc_halo_idx)) * 100.))

                        #print(tot_star_nbs[ihalo]-1,star_nbs[ihalo])
                        star_ids = halo_star_ids[tot_star_nbs[ihalo]-star_nbs[ihalo]:tot_star_nbs[ihalo]]
                        halo_stars = read_specific_stars(star_path, star_ids)
                        
                        stellar_mags=get_star_mags_metals(halo_stars['age'],halo_stars['Z/0.02']*0.02,mag_fct)

                        stellar_fluxes=10**(stellar_mags/-2.5) * halo_stars['mass'] / (1 - eta_sn)

                        tot_stellar_flux = np.sum(stellar_fluxes)

                        if tot_stellar_flux>0:
                                halo_mags[ihalo]=-2.5*np.log10(tot_stellar_flux)

                        #print(stellar_mags, stellar_fluxes, halo_mags[ihalo])
                
                comm.Barrier()

                #print(halo_mags, (halo_mags).max(), (halo_mags[halo_mags>-99]).min(), np.mean(halo_mags[halo_mags>-99]))
                
                halo_mags = sum_arrays_to_rank0(comm, halo_mags)

                comm.Barrier()
                
                if rank == 0:

                        print("Found magnitudes for snapshot %s"%out_nb)

                        #halo_mags[halo_mags==0]=-99
                        print(halo_mags, (halo_mags).max(), (halo_mags[halo_mags>-99]).min(), np.mean(halo_mags[halo_mags>-99]))
                        
                        

                        with h5py.File(out_name,'w') as out_halos:

                                out_halos.create_dataset("ID",data=halos["ids"] ,dtype=np.int64, compresstion='lzf')
                                out_halos.create_dataset("MAB1600",data=halo_mags ,dtype=np.float32, compresstion='lzf')

            
"""
Main body
"""


if __name__ =='__main__':

        Arg_parser = argparse.ArgumentParser('Use associations to build list of halo magnitudes and halo stellar masses')

        Arg_parser.add_argument('nb',metavar='nsnap',type=int,help='snap number string, give as "XXX" so that 00001 is "001" and 00111 is "111"')
        Arg_parser.add_argument('ldx',metavar='ldx',type=int,help='box size in cells')
        Arg_parser.add_argument('path',metavar='path',type=str,help='path to sim')
        Arg_parser.add_argument('simname',metavar='simname',type=str,help='sim name (will be used to create dirs)')
        Arg_parser.add_argument('--rtwo_fact',metavar='rtwo_fact',type=float,help='1.0 -> associate stellar particles within 1xR200 for all halos',default=1)        
        Arg_parser.add_argument('--ll',metavar='ll',type=float,help='linking length e.g. 0.2',default=0.2)            
        Arg_parser.add_argument('--assoc_mthd',metavar='assoc_mthd',type=str,help='method for linking stars to fof',default='')
        Arg_parser.add_argument('--overwrite',acttion="store_true",help='rewrite existing files')
        
        args = Arg_parser.parse_args()


        out_nb = args.nb
        ldx=args.ldx
        path=args.path
        sim_name=args.simname
        rtwo_fact=args.rtwo_fact
        ll=args.ll
        assoc_mthd=args.assoc_mthd
        
        get_halo_mags(out_nb,ldx,path,sim_name,rtwo_fact=rtwo_fact,ll=ll,assoc_mthd=assoc_mthd,overwrite=args.overwrite)