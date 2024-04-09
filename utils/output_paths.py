import os
import numpy as np


class dataset:
    def print_param_info(self):
        print("Anaylsis parameters:")
        print(f"   fof linking length : {self.ll:.2f}")
        print(f"   r_gal : {self.r200:.1f}xr200")
        print(f"   r_fesc : {self.rfesc:.1f}xr200")
        print(f"   r_star : {self.rstar:.1f}xr200")
        print(f"   association method : {self.assoc_mthd:s}")
        print(f"   max DTM : {self.max_dtm:.2f}")
        if self.mp:
            print(f"   Using Mei Palanque's segmented catalogues")
        if self.clean:
            print(f"   Each star particle is only counted once")
        if self.neb_cont_file_name is not None:
            print(f"   Nebular continuum file : {self.neb_cont_file_name:s}")

    def __init__(self, **kwargs):
        self.ll = 0.2
        self.r200 = 1.0
        self.rfesc = 1.0
        self.rstar = 1.0
        self.assoc_mthd = "stellar_peak"
        self.clean = True
        self.mp = True
        self.max_dtm = 0.5
        self.neb_cont_file_name = None

        if "ll" in kwargs.keys():
            self.ll = kwargs["ll"]

        if "r200" in kwargs.keys():
            self.r200 = kwargs["r200"]

        if "fesc_rad" in kwargs.keys():
            self.rfesc = kwargs["fesc_rad"]

        if "rstar" in kwargs.keys():
            self.rstar = kwargs["rstar"]

        if "assoc_mthd" in kwargs.keys():
            self.assoc_mthd = kwargs["assoc_mthd"]

        if "clean" in kwargs.keys():
            self.clean = kwargs["clean"]

        if "mp" in kwargs.keys():
            self.mp = kwargs["mp"]

        if "max_DTM" in kwargs.keys():
            self.max_dtm = kwargs["max_DTM"]

        if "neb_cont_file_name" in kwargs.keys():
            self.neb_cont_file_name = kwargs["neb_cont_file_name"]

        self.print_param_info()  # dump info to stdout

    def path_2_params(path):
        # todo read path and fill this object with the contained parameter values
        return ()


def get_fof_suffix(fof_path):
    fof_suffix = ""
    cut_path = np.asarray(fof_path.split("/"))
    print(cut_path)
    find_ll = np.where(["ll_0p" in s for s in cut_path])
    if len(find_ll[0]) == 0:
        return fof_suffix
    fof_suffix = cut_path[find_ll[0][0]]

    return fof_suffix


def ll_to_fof_suffix(ll):
    if type(ll) == float:
        ll = int(ll * 1000)

    return "ll_0p%i" % ll


def get_r200_suffix(rtwo_fact):
    rtwo_suffix = ""

    if rtwo_fact != 1:
        if int(rtwo_fact) - rtwo_fact == 0:
            rtwo_suffix += "%ixr200" % rtwo_fact
        else:
            rtwo_suffix += "%.1fxr200" % rtwo_fact
            rtwo_suffix = rtwo_suffix.replace(".", "p")

    return rtwo_suffix


def get_frad_suffix(frad_fact):
    frad_suffix = ""

    if frad_fact != 1 and frad_fact != 1.0:
        if int(frad_fact) - frad_fact == 0:
            frad_suffix += "frad%i" % frad_fact
        else:
            frad_suffix += "frad%.1f" % frad_fact
            frad_suffix = frad_suffix.replace(".", "p")

    return frad_suffix


def get_suffix(fof_suffix="", rtwo_suffix="", frad_suffix="", mp=False):
    suffix = ""

    if fof_suffix != "":
        suffix += "_%s" % fof_suffix

    if rtwo_suffix != "":
        suffix += "_%s" % rtwo_suffix

    if frad_suffix != "":
        suffix += "_%s" % frad_suffix

    if mp:
        suffix += "_mp"

    return suffix


def check_assoc_keys(assoc):
    possible_keys = ["", "star_barycentre", "stellar_peak", "fof_ctr"]

    found = [assoc == key for key in possible_keys]

    assert np.sum(found) > 0, (
        "error association method unrecognized, possible methods ares %s"
        % possible_keys
    )


def gen_paths(sim_name, out_nb, dataset):
    out = os.path.join(
        "/lustre/orion/proj-shared/ast031/jlewis/", sim_name + "_analysis"
    )

    fof_suffix = ll_to_fof_suffix(dataset.ll)
    rtwo_suffix = get_r200_suffix(dataset.r200)
    frad_suffix = get_frad_suffix(dataset.rfesc)

    assoc_suffix = get_suffix(
        fof_suffix=fof_suffix,
        rtwo_suffix=rtwo_suffix,
        mp=dataset.mp,
    )

    analy_suffix = get_suffix(
        fof_suffix=fof_suffix,
        rtwo_suffix=rtwo_suffix,
        frad_suffix=frad_suffix,
        mp=dataset.mp,
    )

    if dataset.assoc_mthd == "" or dataset.assoc_mthd == "fof_ctr":
        assoc_out = os.path.join(out, ("assoc_halos_%s" % out_nb) + assoc_suffix)
        analy_out = os.path.join(out, ("results_halos_%s" % out_nb) + analy_suffix)

    else:
        assoc_out = os.path.join(
            out, ("assoc_%s_halos_%s" % (dataset.assoc_mthd, out_nb)) + assoc_suffix
        )
        analy_out = os.path.join(
            out, ("results_%s_halos_%s" % (dataset.assoc_mthd, out_nb)) + analy_suffix
        )

    if dataset.rstar != 1:
        analy_out += f"_rstar_0p{int(dataset.rstar*100):02d}"

    if dataset.max_dtm != 0.5:
        analy_out += f"_maxDTM_0p{int(dataset.max_dtm*100):02d}"

    if dataset.clean:
        analy_out += "_clean"

    if dataset.neb_cont_file_name is not None:
        analy_out = os.path.join(
            analy_out, f"_neb_cont_{dataset.neb_cont_file_name.replace('.txt','')}"
        )

    return (out, assoc_out, analy_out, analy_suffix)
