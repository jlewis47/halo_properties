from ..params.params import *

def get_unit_facts(data_type, px_to_m, unit_d, unit_l, unit_t, aexp):

    if data_type=='tau':
        return(px_to_m*sigma_UV*1e6*0.76) #important musnt count the helium...
    elif data_type=='rho':
        return(1e-3*unit_d/Pmass) #H/cm**3
    elif data_type=='rhod':
        return(unit_d) #g/cm^3
    elif data_type=='vvrho':
        return(1e6*1e3*1e-2*unit_l/unit_t/1e3) #m/s #H/m**2/s                                                    
    elif data_type=='temp':
        return(Pmass*(1e-2*unit_l/unit_t)**2/Bmann) #K
    elif data_type=='flux':
        return(aexp**-3.) #m**-2*s**-1                                                                               

    
def convert_temp(temp, rho, xtion):

    return(temp / (rho * (1. + xtion)))

