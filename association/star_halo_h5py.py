'''
------
Read halo and star files and make associations
Saved as binary file that repeats for N halos and another that contains M stars
Associated stars and non associated stars are saved to separate binary files
Two files track each halo star number and 1st star file posittion
A final file saves halo IDs
------

This verstion includes halos that are close to each other and/or without stars
It also tracks stars that don't end up associated, that are saved to separate 'LONE' star files

'''


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from read_phew import read_phew
from tionread_stars import read_all_star_files
from scipy import spatial
import time
import string
import argparse
import os
from utils.functions_latest import *
from utils.utils import *
#from read_fof import o_fof
import h5py
from utils.output_paths import *
from files.wrap_boxes import *
from scipy.stats import binned_statistic_dd

def find_nearby_stars(star_tree, r_px, ctr):

        return(ctr,star_tree.query_ball_point(ctr,r_px))

def find_nearby_stars_recursive_stellar(star_coords,star_masses, star_tree, r_px, ctr, ldx, mvt=0, overstep=False):

        big_ball = star_tree.query_ball_point(ctr,r_px)
        cur_star_coords=star_coords[big_ball,:]
        cur_star_masses=star_masses[big_ball]


        l=len(cur_star_masses)
        if l<1:
                return(ctr,[])
        elif l==1:
                stellar_barycentre=cur_star_coords[0]
        elif not overstep: #if we don't account for cases where halo is on edge of box
                stellar_barycentre=np.average(cur_star_coords,weights=cur_star_masses,axis=0)
        else:
                #print(overstep,r_px)
                #stellar_barycentre=get_ctr_mult(cur_star_coords,pos_vects,cur_star_masses)
                stellar_barycentre=get_ctr_mult_cheap(ctr,cur_star_coords,ldx,cur_star_masses)
                #print(cur_star_coords)
                #print(stellar_barycentre)
                #print(ctr)
                
        loc_mvt=np.linalg.norm([stellar_barycentre-ctr])

        mvt+=loc_mvt

        ##print(mvt,loc_mvt)
        if loc_mvt<0.5 or mvt>r_px:
                return(ctr,big_ball)
        
        else:
                ctr,big_ball=find_nearby_stars_recursive_stellar(star_coords, star_masses, star_tree, r_px, stellar_barycentre, ldx, mvt=mvt)
                return(ctr,big_ball)

def find_max_stellar_recursive(star_coords,star_masses, star_tree, r_px, ctr, ldx, mvt=0, overstep=False):


        
        big_ball = star_tree.query_ball_point(ctr,r_px)
        cur_star_coords=star_coords[big_ball,:]
        cur_star_masses=star_masses[big_ball]                


        n_found=len(cur_star_masses)
        if n_found<1:
                return(ctr,[])

        # if r_px>1:
        #         print('top',np.sum(cur_star_masses))
        #print(cur_star_masses, cur_star_coords)

        
        l=np.ceil(r_px)+2
        bin_size=1.0
        coord_bins=np.arange(-l,l,bin_size)

        #print(coord_bins,coord_bins-0.5*l+ctr[0],coord_bins-0.5*l+ctr[1],coord_bins-0.5*l+ctr[2])
        
        stellar_density_map, edges, numbers=binned_statistic_dd(cur_star_coords,cur_star_masses,'sum',bins=[coord_bins+ctr[0], coord_bins+ctr[1], coord_bins+ctr[2]])



        stellar_max=np.max(stellar_density_map)

        #assert abs(np.sum(stellar_density_map)-np.sum(cur_star_masses))<1e2, print(np.sum(stellar_density_map),np.sum(cur_star_masses))
        
        #print(np.shape(stellar_density_map), np.where(stellar_density_map==stellar_max))


        whs=np.where(stellar_density_map==stellar_max)[0]
        if len(whs)>1:
                stellar_peaks=[wh*bin_size-0.5*l + ctr for wh in whs]
                # mvts=[np.linalg.norm([stellar_peak-ctr]) for stellar_peak in stellar_peaks]
                # whs=whs[np.argmin(mvts)]
                stellar_peak=np.mean(stellar_peaks,axis=0)
                # if r_px>1:
                #         print('issue')
                #         print(stellar_peak,ctr)
                

        else:
                stellar_peak=whs*bin_size-0.5*l + ctr


        
        loc_mvt=np.linalg.norm([stellar_peak-ctr])

        mvt+=loc_mvt


        
        ##print(mvt,loc_mvt)
        if loc_mvt<0.5 or mvt>r_px:

                #if r_px>1:print('done',loc_mvt,ctr,np.sum(cur_star_masses))
                
                return(ctr,big_ball)
        
        else:

                # if r_px>1:
                #         print('bot')
                #         print(ctr, stellar_peak, np.sum(cur_star_masses),  mvt, loc_mvt)
                #         #print(stellar_density_map)
                        
                ctr,big_ball=find_max_stellar_recursive(star_coords, star_masses, star_tree, r_px, stellar_peak, ldx, mvt=mvt)
                return(ctr,big_ball)

        
def find_nearby_stars_wrapper(star_coords, star_masses, star_tree, r_px, ctr, ldx, assoc_mthd, overstep):

        if assoc_mthd=='':
                ctr,bb=find_nearby_stars(star_tree,r_px,ctr)

        elif assoc_mthd=='star_barycentre':
                #pos_vects=gen_pos_vects(ldx)
                ctr,bb=find_nearby_stars_recursive_stellar(np.asarray(star_coords), np.asarray(star_masses), star_tree, r_px, ctr, ldx, 0.0, overstep)
        elif assoc_mthd=='stellar_peak':
                #pos_vects=gen_pos_vects(ldx)
                ctr,bb=find_max_stellar_recursive(np.asarray(star_coords), np.asarray(star_masses), star_tree, r_px, ctr, ldx, 0.0, overstep)


                
        return(ctr,bb)
        
def assoc_stars_to_haloes(out_nb,ldx,path,sim_name,use_fof=True,rtwo_fact=1,fof_path=None,npart_thresh=50,assoc_mthd=''):

        check_assoc_keys(assoc_mthd)
        
        #phew_path='/data2/jlewis/dusts/output_00'+out_nb
        #star_path='/data2/jlewis/dusts/'
        #info_path='/data2/jlewis/dusts/output_00'+out_nb

        output_str='output_%06i'%out_nb
        
        info_path=os.path.join(path,output_str,'group_000001')
        star_path=os.path.join(path,'reduced','stars',output_str)
        #phew_path=os.path.join(path,output_str)

        fof_suffix=get_fof_suffix(fof_path)
        rtwo_suffix=get_r200_suffix(rtwo_fact)
        suffix=get_suffix(fof_suffix,rtwo_suffix)
                        
        Np_tot=ldx**3

        '''Get scale factor and co'''
        (t,a,H0,om_m,om_l,om_k,om_b,unit_l,unit_d,unit_t,l,Lco,L,px_to_m) = get_infos(info_path,
                                                                                           out_nb,
                                                                                      ldx)
        print('using r=%iXr200'%rtwo_fact)

        tstep=0.0 #temporary

        out,assoc_out,analy_out=gen_paths(sim_name,out_nb,suffix,assoc_mthd)
        
        #if no out folder, make it
        if not os.path.isdir((assoc_out)):
            os.makedirs(assoc_out)

        #if no out folder, make it
        if not os.path.isdir((analy_out)):
            os.makedirs(analy_out)

        #Mp=3.*(H0*1e3/(pc*1e6))**2.*(om_m-om_b)*(Lco*pc*1e6)**3./Np_tot/8./np.pi/G/Msol #En Msol 4.04*1e5
        Mp=get_Mp(om_m,om_b,H0,Lco,Np_tot)
        print("Partcle mass is :",Mp)
        
        if not  'Mp' in  os.listdir(out):
                out_Mp = open(os.path.join(out,'Mp'),'wb')
                out_Mp.write(np.float64(Mp))
                out_Mp.close()

        halos=[]
        
        with h5py.File(os.path.join(fof_path,output_str,'haloes_masst.h5'), 'r') as src:

                keys=src['Data'].keys()

                for key in keys:
                        halos.append(src['Data'][key][()])

        halos=np.asarray(halos).T        

        #print(np.log10(np.max(halos[:,1])*Mp),get_r200(np.max(halos[:,1])))

        new_ctrs=np.zeros((len(halos),3))
        
        #trim <npart_thresh particle halos
        if npart_thresh>50.0:halos=halos[halos[:,1]>=npart_thresh,:]

        nhalos=len(halos)
        
        print('Found %i haloes with npart>%i'%(nhalos,max(npart_thresh,50)))

        halo_star_nb=np.zeros(nhalos)
        
        #Stars
        stars=read_all_star_files(star_path)

        families=stars['family']
        
        stars=np.transpose([stars['mass'],stars['x'],stars['y'],stars['z'],stars['age'],stars['Z/0.02']])
        #filter non-stars (->no debris)
        stars=stars[families==2,:]

        halo_stars=[]

        stellar_coords=stars[:,1:4]*(ldx)
        stellar_coords[stellar_coords>ldx]-=ldx
        star_tree = spatial.cKDTree(stellar_coords,boxsize=ldx+1e-6)
        
        #Local star IDs for creating list of 'lone stars'
        star_ids = np.arange(len(stars))

        halo_star_ids=[]
        
        size=np.asarray(np.int32(len(halos)))

        stt = time.time()

        tot_nb_stars=0

        #Associattion and output

        out_halos = open(os.path.join(assoc_out,('assoc_halos_%s' %out_nb)+suffix),'wb')
        out_halosST = open(os.path.join(assoc_out,('assoc_halosST_%s' %out_nb)+suffix),'wb')
        out_halosID = open(os.path.join(assoc_out,('assoc_halosID_%s' %out_nb)+suffix),'wb')
        out_stars = open(os.path.join(assoc_out,('assoc_stars_%s' %out_nb)+suffix),'wb')
        out_halosST_IDs = open(os.path.join(assoc_out,('assoc_starsIDs_%s' %out_nb)+suffix),'wb')
        out_lone_stars = open(os.path.join(assoc_out,('assoc_lone_stars_%s' %out_nb)+suffix),'wb')

        np.save(out_halos,size)

        not_lone_star_ids = []

        r_pxs = get_r200(halos[:,1])*rtwo_fact


        np.save(out_halosID,np.float64(halos[:,0]))

        new_halo_tab=np.c_[halos[:,1:],r_pxs]


        out_halosID.close()        

        print('Associattion starting')
        
        for halo_nb,halo in enumerate(halos[:]):

            #print(r_px)

            r_px=r_pxs[halo_nb]

            ctr_vanilla = halo[2:5] #ctr without rounding
            ctr = halo[2:5]*(ldx)

            overstep=np.any(ctr+r_px>ldx) or np.any(ctr-r_px<0)
            
            #Find nearby stars
            new_ctr, big_ball=find_nearby_stars_wrapper(stellar_coords, stars[:,0], star_tree, r_px, ctr, ldx, assoc_mthd, overstep=overstep)
            
            nb_stars = np.count_nonzero(big_ball)
            
            if nb_stars==0:continue
            
            new_ctrs[halo_nb,:]=new_ctr[:]
            
            halo_stars.extend(stars[big_ball,:])
            
            halo_star_ids.extend(star_ids[big_ball])
            
            halo_star_nb[halo_nb]=nb_stars

            #print(halo_nb,nb_stars)
            
            tot_nb_stars+=nb_stars
            
        print('Associattion Finished')


        np.save(out_halos,np.c_[new_halo_tab,new_ctrs])
        out_halos.close()
        
        np.save(out_stars,halo_stars)
        np.save(out_halosST_IDs,halo_stars)        
        
        out_stars.close()
        out_halosST_IDs.close()        

        np.save(out_halosST, halo_star_nb)
        out_halosST.close()

        print('Saving lonely stars')
        
        #Write unassigned stars
        not_lone_star_ids=set(halo_star_ids)
        lone_star_ids=set(star_ids)-not_lone_star_ids        
        
        lone_stars = stars[list(lone_star_ids),:]
        np.save(out_lone_stars,np.float32(lone_stars[:,:]))
        out_lone_stars.close()

        print(len(not_lone_star_ids),len(lone_star_ids))

        print('Done')

"""
Main body
"""


if __name__ =='__main__':

        Arg_parser = argparse.ArgumentParser('Associate stars and halos in full simulattion')

        Arg_parser.add_argument('nb',metavar='nsnap',type=int,help='snap number string, give as "XXX" so that 00001 is "001" and 00111 is "111"')
        Arg_parser.add_argument('ldx',metavar='ldx',type=int,help='box size in cells')
        Arg_parser.add_argument('path',metavar='path',type=str,help='path to sim')
        Arg_parser.add_argument('simname',metavar='simname',type=str,help='sim name (will be used to create dirs)')
        Arg_parser.add_argument('fofpath',metavar='fofpath',type=str,help='folder for fof masst hdf5 files')
        Arg_parser.add_argument('--rtwo_fact',metavar='rtwo_fact',type=float,help='1.0 -> associate stellar particles within 1xR200 for all haloes',default=1)
        Arg_parser.add_argument('--npart_thresh',metavar='npart_thresh',type=float,help='dark matter particle number threshold for halos',default=50)
        Arg_parser.add_argument('--assoc_mthd',metavar='assoc_mthd',type=str,help='method for linking stars to fof',default='')                        

        
        args = Arg_parser.parse_args()


        out_nb = args.nb
        ldx=args.ldx
        path=args.path
        sim_name=args.simname
        fof_path=args.fofpath
        rtwo_fact=args.rtwo_fact
        npart_thresh=args.npart_thresh
        assoc_mthd=args.assoc_mthd
        
        assoc_stars_to_haloes(out_nb,ldx,path,sim_name,fof_path=fof_path,npart_thresh=npart_thresh,rtwo_fact=rtwo_fact,assoc_mthd=assoc_mthd)
    