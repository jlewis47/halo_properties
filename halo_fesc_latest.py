'''

'''


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from read_radgpu import o_rad_cube_big
from read_stars import read_stars
from scipy import spatial
import time
import string
import argparse
import os
from read_assoc_latest import *
from read_fullbox_big import *
from functions_latest import *
import healpy as hp
from dust_opacity import *
from constants_latest import *
#from wrap_boxes import *
from output_paths import *



def compute_fesc(out_nb,ldx,path,sim_name):


        
        info_path=os.path.join(path,'output_00'+out_nb)
        data_pth_fullres=path
        phew_path=os.path.join(path,'output_00'+out_nb)
        data_pth_assoc=os.path.join(assoc_path,sim_name,'assoc'+out_nb)

        plt.rcParams.update({'font.size': 18})   




        out = os.path.join(analy_path,sim_name)

        #Get scale factor and co
        '''Get scale factor and co'''
        (t,a,H0,om_m,om_l,om_k,om_b,unit_l,unit_d,unit_t,l,Lco,L,px_to_m) = get_infos_no_t(info_path,
                                                                                           out_nb,
                                                                                           ldx)
        tstep=0 #temp need this >0!

        with open(os.path.join(out,'Mp'), 'rb') as mass_file :
            Mp = np.fromfile(mass_file,dtype=np.float64)
        print(Mp)

        dist_obs=345540.98618977674 #distance to obs point from box (0,0,0); in number of cells for z=6


        mags,xis,contbetalow,contbetahigh,beta,metal_bins,age_bins=get_mag_tab_BPASSV221_betas()

        upper=27
        grid = np.mgrid[0:upper,0:upper,0:upper]/float(upper-1)

        print('Getting Phew and stars')
        idxs,star_idxs,phew_tot_star_nb,phew_star_nb,phew_tab,stars,lone_stars = read_assoc(out_nb,sim_name)




        prog_bar0=['[',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',']']
        prog_bar=prog_bar0
        nb_halo = np.shape(phew_tab)[0]

        ran = [-ldx,0,ldx]
        pos_vects = np.asarray([[i,j,k] for k in ran for j in ran for i in ran])

        phew_sumd_fl,phew_nb_cells = np.zeros(3),np.zeros(3)


        #Conversion factors
        tau_fact = px_to_m*sigma_UV*1e6
        rho_fact=1e-3*unit_d/Pmass#unit_d*1e-3*(1e2)**3. #H/cm**3
        dust_fact=unit_d #g/cm^3
        vvrho_fact=1e6*1e3*1e-2*unit_l/unit_t/1e3 #m/s #H/m**2/s
        temp_fact=Pmass*(1e-2*unit_l/unit_t)**2/Bmann #K
        fl_fact=(a**-3.) #m**-2*s**-1


        stt = time.time()



        nside=8 #2 or 4 else wise too costly
        npix=hp.nside2npix(nside)
        pix=np.arange(npix)
        thes,phis=hp.pix2ang(nside,pix)
        Xvects,Yvects,Zvects=hp.pix2vec(nside,pix) #with radius = 1

        hp_surf_elem=4.*np.pi*px_to_m**2 #in m^2.dx^-2


        #print(len(phis))

        rad_res=0.1 #in units of dx

        xbins=np.arange(8,12.1,0.05)
        ybins=np.arange(-2,0.1,0.05)
        nbs_tot=np.zeros((len(xbins)-1,len(ybins)-1))


        limit_r=phew_tab[:,-1]+1
        sample_r=do_half_round(limit_r)


        M,pos_nrmd = phew_tab[:,0],(phew_tab[:,1:4])
        pos =do_half_round((ldx)*pos_nrmd) #was np.int16

        #(0,0,0) px location of 512**3 cube within whole data set


        ctr_bxd=pos

        lower_bounds=np.int32(ctr_bxd-sample_r[:,np.newaxis])
        upper_bounds=np.int32(ctr_bxd+sample_r[:,np.newaxis])

        under=lower_bounds<=0
        over=upper_bounds>=511

        outside=[under,over]

        large_box=np.any(outside)



        #Get data
        #fl= o_rad_cube_big(data_pth_rad,2,f_side,subcube_nb)
        # if tst :nrg = o_rad_cube_big(data_pth_rad,1,f_side,subcube_nb)
        xion=o_data(os.path.join(data_pth_fullres,'xion_00'+out_nb))
        rho=o_data(os.path.join(data_pth_fullres,'rho_00'+out_nb))
        dust=o_data(os.path.join(data_pth_fullres,'dust_00'+out_nb))
        metals=o_data(os.path.join(data_pth_fullres,'Z_00'+out_nb)) #absolute
        # temp =  o_fullbox_big(data_pth_fullres,'temp',512,512,subcube_nb)
        # temp = temp/(rho*(1.+xion))
        #rho = rho
        # vx =  o_fullbox_big(data_pth_fullres,'vx',512,512,subcube_nb)
        # vy =  o_fullbox_big(data_pth_fullres,'vy',512,512,subcube_nb)
        # vz =  o_fullbox_big(data_pth_fullres,'vz',512,512,subcube_nb)    
        # vv = np.asarray([vx,vy,vz]) 
        # vvrho=vv*rho


        #This bit makes an array dilate_fact times bigger than the RAMSES data and fills using the periodicity of the data that way we don't care about being close to the edge with r200 (memory inefficient but fast, only works for one big chuck .... best use with ldx<=1024)

        dilate_fact=1.05
        rescale=(dilate_fact-1)/2.
        delta=int(rescale*ldx)

        #=do_half_round(pos_nrmd*ldx+delta)


        lower_bounds=np.int32(pos-sample_r[:,np.newaxis])+delta
        upper_bounds=np.int32(pos+sample_r[:,np.newaxis])+delta
        ctr_bxd=ctr_bxd+delta
        ctr_big_box=np.int16(pos+delta)

        big_xion=np.zeros((int(ldx*dilate_fact),int(ldx*dilate_fact),int(ldx*dilate_fact)),dtype=np.float64)
        big_rho=np.zeros((int(ldx*dilate_fact),int(ldx*dilate_fact),int(ldx*dilate_fact)),dtype=np.float64)              
        big_metals=np.zeros((int(ldx*dilate_fact),int(ldx*dilate_fact),int(ldx*dilate_fact)),dtype=np.float64)
        big_dust=np.zeros((int(ldx*dilate_fact),int(ldx*dilate_fact),int(ldx*dilate_fact)),dtype=np.float64)                 

        centre_slice=slice(int(rescale*ldx),int((1+rescale)*ldx))

        big_rho[centre_slice,centre_slice,centre_slice]=np.copy(rho)
        big_xion[centre_slice,centre_slice,centre_slice]=np.copy(xion)
        big_metals[centre_slice,centre_slice,centre_slice]=np.copy(metals)
        big_dust[centre_slice,centre_slice,centre_slice]=np.copy(dust)

        #big_dust=big_dust #g/cm^3


        sides=[-1,0,1]

        def put_and_grab(ind,delta,dim):
            """
            Ind gives pos near side of box
            side of larger box is smaller box side +2*delta
            Returns slices necessary to repeat box sides in larger array
            """


            if ind==0 :
                put=slice(delta,dim+delta)
                grab=slice(0,dim)
            elif ind==1 :
                put=slice(dim+delta,dim+2*delta)
                grab=slice(0,delta)
            elif ind==-1 :
                put=slice(0,delta)
                grab=slice(dim-delta,dim)

            return(grab,put)

        for i in sides:
            for j in sides:
                for k in sides: 

                    xgrab,xput=put_and_grab(i,delta,ldx)
                    ygrab,yput=put_and_grab(j,delta,ldx)
                    zgrab,zput=put_and_grab(k,delta,ldx)            


                    big_rho[xput,yput,zput]=np.copy(rho[xgrab,ygrab,zgrab])
                    big_xion[xput,yput,zput]=np.copy(xion[xgrab,ygrab,zgrab])
                    big_metals[xput,yput,zput]=np.copy(metals[xgrab,ygrab,zgrab])            
                    big_dust[xput,yput,zput]=np.copy(dust[xgrab,ygrab,zgrab])
        
        

        halo_ray_Tr=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_ray_Tr_dust_SMC=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_ray_Tr_dust_LMC=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_ray_Tr_dust_MW=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)        
        halo_mags=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_mags_ext_SMC=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_mags_ext_LMC=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_mags_ext_MW=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)        
        halo_SFR10s=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_stellar_mass=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_emissivity=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_youngest=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_oldest=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_betas=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_betas_with_dust_SMC=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_betas_with_dust_MW=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_betas_with_dust_LMC=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)        



        for ind,halo in enumerate(phew_tab):



            r_px=halo[-1]

            r=r_px*px_to_m #Convert to meters
            surf_elem=(hp_surf_elem*r_px**2/float(npix))






            sm_rho = big_rho[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]
            sm_dust = big_dust[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]
            sm_metals = big_metals[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]
            sm_xion = big_xion[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]

            sm_xHI=1-sm_xion

            sm_tau = (sm_xHI*sm_rho)*(rho_fact*tau_fact) #includes partial dx

            sm_tau_dust_1600_SMC = (sm_dust)*(dust_1600_opacity_SMC*px_to_m*100.0)*dust_fact
            sm_tau_dust_LyC_SMC = (sm_dust)*(dust_LyC_opacity_SMC*px_to_m*100.0)*dust_fact

            sm_tau_dust_1600_MW = (sm_dust)*(dust_1600_opacity_MW*px_to_m*100.0)*dust_fact
            sm_tau_dust_LyC_MW = (sm_dust)*(dust_LyC_opacity_MW*px_to_m*100.0)*dust_fact

            sm_tau_dust_1600_LMC = (sm_dust)*(dust_1600_opacity_LMC*px_to_m*100.0)*dust_fact
            sm_tau_dust_LyC_LMC = (sm_dust)*(dust_LyC_opacity_LMC*px_to_m*100.0)*dust_fact
            
            
            rs=np.arange(0,2*r_px,rad_res) #so we do edge cases properly

            Rs,Phis=np.meshgrid(rs,phis)#for healpix
            Rs,Thes=np.meshgrid(rs,thes)#for healpix
            #Rs,Phis,Thes=np.meshgrid(rads,phis,thes) #for a meshgrid generated grid
            Xs,Ys,Zs=sph_2_cart(rs,Phis,Thes)        


            #X_circ,Y_circ,Z_circ=sph_2_cart(r_px,phis,thes)

            #If there aren't any stars : no need to calculate emissivities or star formation stuff
            if phew_star_nb[ind]!=0 :


                    #Get stars for halo from list of associated stars
                    cur_stars=stars[phew_tot_star_nb[ind]:phew_tot_star_nb[ind]+phew_star_nb[ind],:]

                    halo_stellar_mass[ind]=np.sum(cur_stars[:,0])




                    halo_oldest[ind]=np.max(cur_stars[:,-2])
                    halo_youngest[ind]=np.min(cur_stars[:,-2])



                    young=cur_stars[:,-2]<10.
                    halo_SFR10s[ind]=np.sum(cur_stars[young,0])/10./1e6 #Msun/Myr #np.sum(cur_stars[active,0]*active_fraction)/(10*1e6)



                    halo_fluxes=cur_stars[:,0]*10**(-get_star_mags_metals(cur_stars[:,-2],cur_stars[:,-1]*0.02,mags,age_bins,metal_bins)/2.5)

                    cur_star_emissivity=get_star_xis_metals(cur_stars[:,-2],cur_stars[:,-1]*0.02,xis,age_bins,metal_bins)*cur_stars[:,0] #ph/s


                    low_conts=get_star_mags_metals(cur_stars[:,-2],cur_stars[:,-1]*0.02,contbetalow,age_bins,metal_bins)
                    high_conts=get_star_mags_metals(cur_stars[:,-2],cur_stars[:,-1]*0.02,contbetahigh,age_bins,metal_bins)

                    delta_lamb=np.log10(2621./1492.)

                    halo_betas[ind]=np.log10(np.sum(high_conts*cur_stars[:,0])/np.sum(low_conts*cur_stars[:,0]))/delta_lamb

                    halo_emissivity[ind]=np.sum(cur_star_emissivity) #ph/s

                    halo_mags[ind]=-2.5*np.log10(np.nansum(halo_fluxes)) 

                    emissivity_box = np.zeros_like(sm_rho,dtype=np.float64)
                    #stellar_box_mass = np.zeros_like(sm_rho,dtype=np.float64)

                    #If only one cell (possible!) then we need to special take care as functions arent all built for that
                    if  np.shape(emissivity_box)==1:

                        emissivity_box[:]=1
                        #stellar_box_mass[:]=1
                        
                        #star per star

                        #dust_trans=np.ones_like(halo_fluxes,dtype=np.float32)

                        for star_nb,star in enumerate(cur_stars):


                            sm_ctr = np.int32(star[1:4]*(ldx)-[lower_bounds[ind,0],lower_bounds[ind,1],lower_bounds[ind,2]])+0.5

                            halo_ray_Tr[ind]+=(sum_over_rays_bias(sm_tau,sm_ctr,r_px,rad_res,Xs,Ys,Zs))*(star[0])*cur_star_emissivity[star_nb]
                            halo_ray_Tr_dust_SMC[ind]+=(sum_over_rays_bias(sm_tau_dust_LyC_SMC,sm_ctr,r_px,rad_res,Xs,Ys,Zs))*cur_star_emissivity[star_nb]
                            halo_ray_Tr_dust_LMC[ind]+=(sum_over_rays_bias(sm_tau_dust_LyC_SMC,sm_ctr,r_px,rad_res,Xs,Ys,Zs))*cur_star_emissivity[star_nb]
                            halo_ray_Tr_dust_MW[ind]+=(sum_over_rays_bias(sm_tau_dust_LyC_SMC,sm_ctr,r_px,rad_res,Xs,Ys,Zs))*cur_star_emissivity[star_nb]                            
                            dust_taus[star_nb]=star_path_cheap(sm_ctr,sm_tau_dust_1600_SMC,2*halo[-1])


                        halo_ray_Tr[ind]=halo_ray_Tr[ind]/halo_emissivity[ind]

                        halo_ray_Tr_dust_SMC[ind]=halo_ray_Tr_dust_SMC[ind]/halo_emissivity[ind]
                        halo_ray_Tr_dust_MW[ind]=halo_ray_Tr_dust_MW[ind]/halo_emissivity[ind]
                        halo_ray_Tr_dust_LMC[ind]=halo_ray_Tr_dust_LMC[ind]/halo_emissivity[ind]                        

                        halo_mags_ext_SMC[ind]=-2.5*np.log10(np.nansum(halo_fluxes*np.exp(-dust_taus)))
                        halo_mags_ext_LMC[ind]=-2.5*np.log10(np.nansum(halo_fluxes*np.exp(-dust_taus*dust_1600_opacity_LMC/dust_1600_opacity_SMC)))
                        halo_mags_ext_MW[ind]=-2.5*np.log10(np.nansum(halo_fluxes*np.exp(-dust_taus*dust_1600_opacity_MW/dust_1600_opacity_SMC)))

                        halo_betas_with_dust[ind]=np.log10(np.sum(high_conts*cur_stars[:,0]*np.exp(-dust_taus*dust_2621_opacity_SMC/dust_1500_opacity_SMC))/np.sum(low_conts*cur_stars[:,0]*np.exp(-dust_taus*dust_1492_opacity_SMC/dust_1500_opacity_SMC)))/delta_lamb


                    #if more than one cell in sm_data stuff
                    else:


                            

                        fracd_stars = np.copy(cur_stars)
                        fracd_stars[:,1:4] = np.asarray(fracd_stars[:,1:4]*(ldx)+delta)

                        fracd_stars[:,1:4] += pos_vects[np.argmin(get_mult_27(pos[ind],fracd_stars[:,1:4]-delta,pos_vects),axis=1)]
         
                        
                        #basically we get the indices of stars for a position histogram
                        sm_xgrid=np.arange(lower_bounds[ind,0],upper_bounds[ind,0],1)
                        star_sm_posx=np.digitize(fracd_stars[:,1],sm_xgrid)-1

                        sm_ygrid=np.arange(lower_bounds[ind,1],upper_bounds[ind,1],1)
                        star_sm_posy=np.digitize(fracd_stars[:,2],sm_ygrid)-1

                        sm_zgrid=np.arange(lower_bounds[ind,2],upper_bounds[ind,2],1)
                        star_sm_posz=np.digitize(fracd_stars[:,3],sm_zgrid)-1                      

                        #print(fracd_stars[:,3],sm_zgrid)

                        #Using indices we can sum up all the emissities in every cell of our halo
                        for istar,star in enumerate(fracd_stars):

                                emissivity_box[star_sm_posz[istar],star_sm_posy[istar],star_sm_posx[istar]]+=cur_star_emissivity[istar]




                                
                        emissivity_box = np.float64(emissivity_box)



                        smldx=np.shape(emissivity_box)[0]
                        xind,yind,zind=np.mgrid[0:smldx,0:smldx,0:smldx]                        
                        
                        cond=(emissivity_box!=0)*(np.linalg.norm([xind-0.5*smldx+0.5,yind-0.5*smldx+0.5,zind-0.5*smldx+0.5],axis=0)<r_px) #need to check that cell centre is in r200 even if stars won't be outside of r200, ok but does this create weird values ??? Seems to fuck up hist plot of fesc=f(M) ... !


                        cells_w_stars=emissivity_box[cond]/np.sum(emissivity_box[cond])
                        xind,yind,zind=xind[cond],yind[cond],zind[cond]

                        dust_taus=np.zeros_like(emissivity_box,dtype=np.float32)
                        
                        dust_trans_SMC=np.ones_like(emissivity_box,dtype=np.float32)
                        dust_trans_LMC=np.ones_like(emissivity_box,dtype=np.float32)
                        dust_trans_MW=np.ones_like(emissivity_box,dtype=np.float32)                        

                            
                        #instead of doing star per star for fesc, fesc_dust, extinction, we loop over halo cells with stellar particles
                        for icell,(cell_w_stars,x_cell,y_cell,z_cell) in enumerate(zip(cells_w_stars,xind,yind,zind)):
                            sm_ctr=np.asarray([x_cell,y_cell,z_cell])+.5

                            halo_ray_Tr[ind]+=(sum_over_rays_bias(sm_tau,sm_ctr,r_px,rad_res,Xs,Ys,Zs))*cell_w_stars
                            halo_ray_Tr_dust_SMC[ind]+=(sum_over_rays_bias(sm_tau_dust_LyC_SMC,sm_ctr,r_px,rad_res,Xs,Ys,Zs))*cell_w_stars
                            halo_ray_Tr_dust_LMC[ind]+=(sum_over_rays_bias(sm_tau_dust_LyC_LMC,sm_ctr,r_px,rad_res,Xs,Ys,Zs))*cell_w_stars
                            halo_ray_Tr_dust_MW[ind]+=(sum_over_rays_bias(sm_tau_dust_LyC_MW,sm_ctr,r_px,rad_res,Xs,Ys,Zs))*cell_w_stars                            



                            big_box_sm_ctr=np.asarray([sm_xgrid[x_cell],sm_ygrid[y_cell],sm_zgrid[z_cell]])
                            
                            dust_taus[z_cell,y_cell,x_cell]=star_path(big_box_sm_ctr,dist_obs,big_dust,dust_1600_opacity_SMC,2*halo[-1],px_to_m*100,ldx)*dust_fact
                  
                        
                        #now we can use our indices again to get the proper tau/trans for every star : SO MUCH MUCH MUCH MUCH FASTER !

                        star_taus=dust_taus[star_sm_posz,star_sm_posy,star_sm_posx]
                        
                        star_trans_SMC=np.exp(-star_taus)
                        star_trans_LMC=np.exp(-star_taus*dust_1600_opacity_LMC/dust_1600_opacity_SMC)
                        star_trans_MW=np.exp(-star_taus*dust_1600_opacity_MW/dust_1600_opacity_SMC)

                        
                        halo_mags_ext_SMC[ind]=-2.5*np.log10(np.nansum(halo_fluxes*star_trans_SMC))
                        halo_mags_ext_LMC[ind]=-2.5*np.log10(np.nansum(halo_fluxes*star_trans_LMC))
                        halo_mags_ext_MW[ind]=-2.5*np.log10(np.nansum(halo_fluxes*star_trans_MW))

                        #divisions here because I didn't want 6x star_taus arrays ...
                        
                        halo_betas_with_dust_SMC[ind]=np.log10(np.sum(high_conts*cur_stars[:,0]*np.exp(-star_taus*dust_2621_opacity_SMC/dust_1600_opacity_SMC))/np.sum(low_conts*cur_stars[:,0]*np.exp(-star_taus*dust_1492_opacity_SMC/dust_1600_opacity_SMC)))/delta_lamb


                        #... so dividing by the SMC values is OK because star_taus is defined with SMC values
                        
                        halo_betas_with_dust_LMC[ind]=np.log10(np.sum(high_conts*cur_stars[:,0]*np.exp(-star_taus*dust_2621_opacity_LMC/dust_1500_opacity_SMC))/np.sum(low_conts*cur_stars[:,0]*np.exp(-star_taus*dust_1492_opacity_LMC/dust_1500_opacity_SMC)))/delta_lamb

                        halo_betas_with_dust_MW[ind]=np.log10(np.sum(high_conts*cur_stars[:,0]*np.exp(-star_taus*dust_2621_opacity_MW/dust_1500_opacity_SMC))/np.sum(low_conts*cur_stars[:,0]*np.exp(-star_taus*dust_1492_opacity_MW/dust_1500_opacity_SMC)))/delta_lamb                        



        idx = np.copy(idxs)
        mass,x,y,z = np.transpose(phew_tab[:,:-1])

        print('Writing data file')

        dict_keys=['idx','M','x','y','z','fesc_ray','fesc_ray_dust_SMC','fesc_ray_dust_LMC','fesc_ray_dust_MW','mag','mag_ext_SMC','mag_ext_LMC','mag_ext_MW','betas','betas_w_dust_SMC','betas_w_dust_LMC','betas_w_dust_MW','SFR10','StelM','oldst','yngst','halo_emissivity']

        write_data=np.transpose([idx,
                                    mass,
                                    x,
                                    y,
                                    z,
                                    halo_ray_Tr,
                                    halo_ray_Tr_dust_SMC,
                                    halo_ray_Tr_dust_LMC,
                                    halo_ray_Tr_dust_MW,                                 
                                    halo_mags,
                                    halo_mags_ext_SMC,
                                    halo_mags_ext_LMC,
                                    halo_mags_ext_MW,                                 
                                    halo_betas,
                                    halo_betas_with_dust_SMC,
                                    halo_betas_with_dust_LMC,
                                    halo_betas_with_dust_MW,                                 
                                    halo_SFR10s,
                                    halo_stellar_mass,
                                    halo_oldest,
                                    halo_youngest,
                                    halo_emissivity])
        
        file_bytes = write_data

        assert len(dict_keys)==len(np.transpose(file_bytes)), "mismatch between number of keys and number of data entries"

        #Write our output ... everything before _out_ is read by the read_out function as the key parameter
        with open(os.path.join(out,'fesc_dust_out_'+out_nb+'_0'),'wb') as newFile:
            np.save(newFile,np.int32(len(idx)))
            np.save(newFile,np.int32(len(dict_keys)))
            np.save(newFile,np.float64(a))
            np.save(newFile,dict_keys)
            np.save(newFile,file_bytes)


"""
Main body
"""


if __name__ =='__main__':

        Arg_parser = argparse.ArgumentParser('Associate stars and halos in full simulation')

        Arg_parser.add_argument('nb',metavar='nsnap',type=str,help='snap number string, give as "XXX" so that 00001 is "001" and 00111 is "111"')
        Arg_parser.add_argument('ldx',metavar='ldx',type=str,help='box size in cells')
        Arg_parser.add_argument('path',metavar='path',type=str,help='path to sim')
        Arg_parser.add_argument('simname',metavar='simname',type=str,help='sim name (will be used to create dirs)')

        args = Arg_parser.parse_args()


        out_nb = args.nb
        ldx=int(args.ldx)
        path=args.path
        sim_name=args.simname

    
        compute_fesc(out_nb,ldx,path,sim_name)
