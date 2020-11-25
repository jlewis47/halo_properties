'''
This version doesn't use mass attenuation coefs and dust masses but does as WU+2020

This ONLY computes magnitudes and their extinction

Metal column densities give an extinction value for a LoS using a MW type law
This extinction is then scaled by a relavent factor (LoS DTM VS MW DTM)

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


def UV(wvln,Rv):

    """
    From Cardelli+89
    """
        
    x=1./wvln
    
    Fa=(-0.04473*(x-5.9)**2-0.009779*(x-5.9)**3)*int(x>=5.9)
    Fb=(0.2130*(x-5.9)**2+0.1207*(x-5.9)**3)*int(x>=5.9)
    
    a=1.752-0.316*x-0.104/((x-4.67)**2+0.341)+Fa
    b=-3.090+1.825*x+1.206/((x-4.62)**2+0.263)+Fb
    
    return(a+b/Rv)   

def watson11_Av_law(nZ):

        """
        Takes metal column density (cm^-2) and turns it into an extinction
        based on GRB measurements in the MW
        """
        #print(nZ[nZ>0.0])
        return(nZ/2.2e21)


def klambda_cardetti(wvln):
    """wvln should be in micrometers"""

    assert np.all((wvln>=0.12)*(wvln<0.63)), 'Wavelength not in range, should be >=0.12mim and <0.63 mim'
    
    Rv=4.05
    x=1.0/wvln
    
    return(2.659*(-2.156+1.509*x-0.198*x**2.0+0.11*x**3.0)+Rv)
    
def compute_AV_ext(out_nb,ldx,path,sim_name):


        
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
        tau_fact = px_to_m*sigma_UV*1e6*0.76 #important musnt count the helium...
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


        rho=o_data(os.path.join(data_pth_fullres,'rho_00'+out_nb))
        dust=o_data(os.path.join(data_pth_fullres,'dust_00'+out_nb))
        metals=o_data(os.path.join(data_pth_fullres,'Z_00'+out_nb)) #absolute

        #This bit makes an array dilate_fact times bigger than the RAMSES data and fills using the periodicity of the data that way we don't care about being close to the edge with r200 (memory inefficient but fast, only works for one big chuck .... best use with ldx<=1024)

        dilate_fact=1.05
        rescale=(dilate_fact-1)/2.
        delta=int(rescale*ldx)


        lower_bounds=np.int32(pos-sample_r[:,np.newaxis])+delta
        upper_bounds=np.int32(pos+sample_r[:,np.newaxis])+delta
        ctr_bxd=ctr_bxd+delta
        ctr_big_box=np.int16(pos+delta)

        big_rho=np.zeros((int(ldx*dilate_fact),int(ldx*dilate_fact),int(ldx*dilate_fact)),dtype=np.float64)              
        big_metals=np.zeros((int(ldx*dilate_fact),int(ldx*dilate_fact),int(ldx*dilate_fact)),dtype=np.float64)
        big_dust=np.zeros((int(ldx*dilate_fact),int(ldx*dilate_fact),int(ldx*dilate_fact)),dtype=np.float64)                 

        centre_slice=slice(int(rescale*ldx),int((1+rescale)*ldx))

        big_rho[centre_slice,centre_slice,centre_slice]=np.copy(rho)
        big_metals[centre_slice,centre_slice,centre_slice]=np.copy(metals)
        big_dust[centre_slice,centre_slice,centre_slice]=np.copy(dust)


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
                    big_metals[xput,yput,zput]=np.copy(metals[xgrab,ygrab,zgrab])            
                    big_dust[xput,yput,zput]=np.copy(dust[xgrab,ygrab,zgrab])
        
        


        halo_mags=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_mags_ext_AVlaw_devriendt=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_mags_ext_AVlaw_wu=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)
        halo_stellar_mass=np.zeros((np.shape(phew_tab)[0]),dtype=np.float64)



        for ind,halo in enumerate(phew_tab):



            r_px=halo[-1]

            r=r_px*px_to_m #Convert to meters
            surf_elem=(hp_surf_elem*r_px**2/float(npix))

            sm_rho = big_rho[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]*unit_d
            sm_dust = big_dust[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]*unit_d
            sm_metals = big_metals[lower_bounds[ind,2]:upper_bounds[ind,2],lower_bounds[ind,1]:upper_bounds[ind,1],lower_bounds[ind,0]:upper_bounds[ind,0]]
            
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

                    cur_star_emissivity=get_star_xis_metals(cur_stars[:,-2],cur_stars[:,-1]*0.02,xis,age_bins,metal_bins)*cur_stars[:,0] #ph/s


                    halo_stellar_mass[ind]=np.sum(cur_stars[:,0])


                    metals_LoS=np.zeros(len(cur_stars),dtype=np.float32)
                    metals_LoS_devriendt=np.zeros(len(cur_stars),dtype=np.float32)



                    halo_fluxes=cur_stars[:,0]*10**(-get_star_mags_metals(cur_stars[:,-2],cur_stars[:,-1]*0.02,mags,age_bins,metal_bins)/2.5)

                    halo_mags[ind]=-2.5*np.log10(np.nansum(halo_fluxes)) 

                    emissivity_box = np.zeros_like(sm_rho,dtype=np.float64)
                    metal_LoS_box = np.zeros_like(sm_rho,dtype=np.float64)
                    metal_LoS_box_devriendt = np.zeros_like(sm_rho,dtype=np.float64)
                    DTM_LoS_box = np.zeros_like(sm_rho,dtype=np.float64)                    


                    
                    #If only one cell (possible!) then we need to special take care as functions arent all built for that
                    if  np.shape(emissivity_box)==1:

                        emissivity_box[:]=1
                        #stellar_box_mass[:]=1
                        
                        #star per star

                        #dust_trans=np.ones_like(halo_fluxes,dtype=np.float32)


                        
                        for star_nb,star in enumerate(cur_stars):


                            sm_ctr = np.int32(star[1:4]*(ldx)-[lower_bounds[ind,0],lower_bounds[ind,1],lower_bounds[ind,2]])+0.5

                            metals_LoS[star_nb]=star_path_cheap(sm_ctr,sm_metals*sm_rho,2*halo[-1])*px_to_m*100.0/Pmass/1e3 #cm^-2
                            metals_LoS_devriendt[star_nb]=star_path_cheap(sm_ctr,(sm_metals/0.02)**1.3*sm_rho,2*halo[-1])*px_to_m*100.0/Pmass/1e3/2.1e21
                            DTM_LoS[star_nb]=star_path_cheap_mean(sm_ctr,sm_dust/sm_metals/sm_rho,2*halo[-1])

                        DTM_LoS[np.isinf(DTM_LoS)]=0.0
                        
                        ext_Av_star=watson11_Av_law(metals_LoS/0.02)*DTM_LoS/0.44 # 0.44 is canonical MW value
                        #should the metallicity be solar?
                        ext_A1600_star=ext_Av_star*klambda_cardetti(0.16)/klambda_cardetti(0.551)
 

                        halo_mags_ext_AVlaw_wu[ind]=-2.5*np.log10(np.nansum(halo_fluxes*10**(ext_A1600_star/-2.5)))
                        



                        #as in devriendt+2010
                        tau_dust=UV(1600,3.1)*metals_LoS_devriendt
                        halo_mags_ext_AVlaw_devriendt[ind]=-2.5*np.log10(np.nansum(halo_fluxes*np.exp(-tau_dust)))

                        

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



                            
                        metal_densities=np.zeros_like(emissivity_box,dtype=np.float32)

                            
                        #instead of doing star per star for fesc, fesc_dust, extinction, we loop over halo cells with stellar particles
                        for icell,(cell_w_stars,x_cell,y_cell,z_cell) in enumerate(zip(cells_w_stars,xind,yind,zind)):
                            sm_ctr=np.asarray([x_cell,y_cell,z_cell])+.5


                            #big_box_sm_ctr=np.asarray([sm_xgrid[x_cell],sm_ygrid[y_cell],sm_zgrid[z_cell]])


                            
                            #metal_LoS_box[z_cell,y_cell,x_cell]=star_path_cheap(sm_ctr,sm_metals*sm_rho/Pmass/1e3,2*halo[-1])*px_to_m*100.0 #cm^-2

                            metal_LoS_box_devriendt[z_cell,y_cell,x_cell]=star_path_cheap(sm_ctr,(sm_metals/0.02)**1.3*sm_rho,2*halo[-1])*px_to_m*100.0/2.1e21/Pmass/1e3 #cm^-2 #ala devriendt+2010
                            metal_LoS_box[z_cell,y_cell,x_cell]=star_path_cheap(sm_ctr,sm_metals*sm_rho,2*halo[-1])*px_to_m*100.0/Pmass/1e3 #cm^-2

                            
                            DTM_LoS_box[z_cell,y_cell,x_cell]=star_path_cheap_mean(sm_ctr,sm_dust/sm_metals/sm_rho,2*halo[-1]) #cm^-2
                  
                        
                        #now we can use our indices again to get the proper tau/trans for every star : SO MUCH MUCH MUCH MUCH FASTER !

                        metals_LoS_devriendt=metal_LoS_box_devriendt[star_sm_posz,star_sm_posy,star_sm_posx]
                        metals_LoS=metal_LoS_box[star_sm_posz,star_sm_posy,star_sm_posx]
                    

                        

                        DTM_LoS=DTM_LoS_box[star_sm_posz,star_sm_posy,star_sm_posx]

                                

                        DTM_LoS[np.isinf(DTM_LoS)]=0.0
                        DTM_LoS[np.isnan(DTM_LoS)]=0.0


                        #as in devriendt+2010
                        tau_dust=UV(1600,3.1)*metals_LoS_devriendt
                        halo_mags_ext_AVlaw_devriendt[ind]=-2.5*np.log10(np.nansum(halo_fluxes*np.exp(-tau_dust)))

                        #as in Wu+2020
                        #print(DTM_LoS/0.44)
                        
                        ext_Av_star=watson11_Av_law(metals_LoS/0.02)*DTM_LoS/0.44 # 0.44 is canonical MW value
                        #should the metallicity be solar?
                        ext_A1600_star=ext_Av_star*klambda_cardetti(0.16)/klambda_cardetti(0.551)



                        halo_mags_ext_AVlaw_wu[ind]=-2.5*np.log10(np.nansum(halo_fluxes*10**(ext_A1600_star/-2.5)))
                        







        idx = np.copy(idxs)
        mass,x,y,z = np.transpose(phew_tab[:,:-1])
        print('**************************************************')
        print("%.1f"%(1.0/a-1.0))
        print('Extinctions : ')
        ext_all=halo_mags_ext_AVlaw_wu-halo_mags
        ext_all[np.isinf(ext_all)]=0
        print(np.nanmax(ext_all))
        print(np.nanmean(ext_all))

        
        print('Writing data file')

        dict_keys=['idx','M','x','y','z','mags','mags_ext_AVlaw_devriendt','mags_ext_AVlaw_wu']

        write_data=np.transpose([idx,
                                 mass,
                                 x,
                                 y,
                                 z,
                                 halo_mags,
                                 halo_mags_ext_AVlaw_devriendt,
                                 halo_mags_ext_AVlaw_wu])
        
        file_bytes = write_data

        assert len(dict_keys)==len(np.transpose(file_bytes)), "mismatch between number of keys and number of data entries"

        #Write our output ... everything before _out_ is read by the read_out function as the key parameter
        with open(os.path.join(out,'halo_mag_ext_AVlaw_out_'+out_nb+'_0'),'wb') as newFile:
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
