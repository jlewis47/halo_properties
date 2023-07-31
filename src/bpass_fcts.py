import numpy as np
from scipy.interpolate import interp2d
from ..params.params import eta_sn
import os

def get_mag_tab():

    
    mags,xis,contbetalow,contbetahigh,beta,metal_bins,age_bins=get_mag_tab_BPASSV221_betas()

    return(mags,xis,metal_bins,age_bins)

def get_mag_tab_BPASSV221_betas(file):

    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)

    reading=[] 
    with open(os.path.join(dir_path,"../BPASSV221/%s"%file),'r') as src:

        for line in src:
            if '#' not in str(line):
                tmp=line.split(',')
                tmp[-1]=tmp[-1].strip('\n')
                #print(tmp)
                reading.append(np.float64(tmp))


    age_bins=reading[2]
    metal_bins=reading[3]

    xis=np.vstack(reading[4:17])
    mags=np.vstack(reading[17:30])
    contbetalow=np.vstack(reading[30:43])
    contbetahigh=np.vstack(reading[43:56])
    beta=np.vstack(reading[56:69])

    return(mags,xis,contbetalow,contbetahigh,beta,metal_bins,age_bins)

def get_mag_interp_fct(mag_tab,Agebins,Zbins):

    return(interp2d(np.log10(Agebins),np.log10(Zbins),(mag_tab),kind='linear'))
    
def get_star_mags(star_age,interp_fct):

    """
    return stellar magnitude table 
    outside of parameter range (Age,Z), values are table edges
    CoDaII -> Z=0.001
    """
    
    return(interp_fct(np.log10(star_age),np.log10(0.001)))

def get_star_mags_metals(star_age,star_metal,interp_fct):

    """
    return stellar magnitudes
    outside of parameter range (Age,Z), values are table edges
    """

    ages=np.copy(star_age)                                                                           
    Zs=np.copy(star_metal)                                                                               
    
    ages[ages<1]=1                                                                                   
    ages[ages>1e5]=1e5                                                                               
                                                                                                     
    Zs[Zs<1e-5]=1e-5                                                                                 
    Zs[Zs>0.04]=0.04       
    
    return(np.asarray([interp_fct(np.log10(age),np.log10(Z))[0] for age,Z in zip(ages,Zs)]))

def get_xis_interp_fct(xis_tab,Agebins,Zbins):

    return(interp2d(np.log10(Agebins),np.log10(Zbins),np.log10(xis_tab),kind='linear'))

def get_star_xis_metals(star_age,star_metal,interp_fct):

    """
    return stellar emissivity in ph/s/msol
    outside of parameter range (Age,Z), values are table edges
    """

    ages=np.copy(star_age)                                                                           
    Zs=np.copy(star_metal)                                                                               
    
    ages[ages<1]=1                                                                                   
    ages[ages>1e5]=1e5                                                                               
                                                                                                     
    Zs[Zs<1e-5]=1e-5                                                                                 
    Zs[Zs>0.04]=0.04       

    return(np.asarray([interp_fct(np.log10(age),np.log10(Z))[0] for age,Z in zip(ages,Zs)]))    

def get_SFR(stars,age_lim):
    """
    Return avg SFR (msol.yr^-1) over last "age_lim" Myrs
    stars is rows of (mass,x,y,z,age)
    """

    underage_stars = stars["age"]<age_lim
    formd_mass=0
    
    if np.sum(underage_stars)>0:
        formd_mass = stars["mass"][underage_stars]

    return(np.sum(formd_mass) / (1 - eta_sn) / (age_lim*1e6))

def get_lum(stars,lum,age_lim):
    """
    Return luminosity (ph.s^-1) over last "age_lim" Myrs
    """
    underage_stars = np.ones_like(lum)==1.0
    if age_lim>0:
        underage_stars = stars["age"]<age_lim

    return(np.sum(np.float64(lum[underage_stars])))    
    # return(np.sum(np.float64(emissivity[underage_stars] * stars["mass"][underage_stars] / (1 - eta_sn))))    

def bin_star_info(halo_SFRs, halo_Lintrs, cur_stars, cur_star_lum, bins):

    for ilim, lim in enumerate(bins):

        halo_Lintrs[ilim] = get_lum(cur_stars, cur_star_lum, lim)           

    for ilim, lim in enumerate(bins[1:]):
        
        halo_SFRs[ilim] = get_SFR(cur_stars, lim) 

    return(halo_SFRs, halo_Lintrs)



def comp_betas(Mst, high_flux, low_flux, tau_dust912, coef_att):

    if coef_att.Kappa912>0.0:
        betas = (np.log10(np.sum(Mst * high_flux * np.exp(-tau_dust912 * coef_att.Kappa2500 / coef_att.Kappa912))
        / np.sum(Mst * low_flux * np.exp(-tau_dust912 * coef_att.Kappa1500 / coef_att.Kappa912)))
        / np.log10(2500. / 1500.))
    else:
        betas = (np.log10(np.sum(Mst * high_flux)
        / np.sum(Mst * low_flux ))
        / np.log10(2500. / 1500.))

    return(betas)


def comp_betas_indv(Mst, high_flux, low_flux, tau_dust912, coef_att):

    """Don't sum so get one beta for each star
    """

    if coef_att.Kappa912>0.0:
        betas = (np.log10((Mst * high_flux * np.exp(-tau_dust912 * coef_att.Kappa2500 / coef_att.Kappa912))
        /(Mst * low_flux * np.exp(-tau_dust912 * coef_att.Kappa1500 / coef_att.Kappa912)))
        / np.log10(2500. / 1500.))
    else:
        betas = (np.log10((Mst * high_flux)
        /(Mst * low_flux ))
        / np.log10(2500. / 1500.))

    return(betas)
