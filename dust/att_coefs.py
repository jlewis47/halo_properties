import numpy as np
from halo_properties.dust.read_draine_tabs import *
from scipy.optimize import curve_fit as cvf
import os


def linr(x, a, b):
    return a * x + b


# att coef type
class att_coefs:
    def __init__(self, name, low, high, ionising, mean_ionising, sixteen=None):
        self.name = name

        self.Kappa912 = ionising
        self.Kappa611 = mean_ionising
        self.Kappa1500 = low
        self.Kappa2500 = high

        if sixteen != None:
            self.Kappa1600 = sixteen

    def guess_wvlngth(self, wvlgnth):
        popts, pcov = cvf(linr, [1500, 2500], [self.Kappa1500, self.Kappa2500])

        print("sterr is %e" % (np.sqrt(np.trace(pcov**2))))

        return linr(wvlgnth, popts[0], popts[1])


def get_dust_att_files():
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)

    att_file_path = os.path.join(dir_path, "../dust_files")
    dust_files = os.listdir(att_file_path)
    # print(dust_files)

    # print(att_file_path)

    files = [os.path.join(att_file_path, f) for f in dust_files]

    return files


def get_dust_att_keys():
    keys = ["no_dust"]

    files = get_dust_att_files()

    for f in files:
        keys.append(f.split("/")[-1])

    return keys


def att_coef_draine_file(path):
    tab = read_draine_tab(path)
    return att_coefs(
        path.split("/")[-1],
        kappa_wvlngth(tab, 1500.0),
        kappa_wvlngth(tab, 2500.0),
        kappa_wvlngth(tab, 912.0),
        kappa_wvlngth(tab, 611.0),
        sixteen=kappa_wvlngth(tab, 1600.0),
    )
