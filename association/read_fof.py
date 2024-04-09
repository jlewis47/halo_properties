import numpy as np
import os
import h5py

# from tionread_stars import read_all_star_files


# data_pth = '/data/ngillet/titan/snap_Lorenzo/B4_256_CoDa/output_00022/'


def o_fof(data_pth):
    """
    First run add total halo number from each masst file
    Then build data array
    Then perform another run to get data
    """

    # Get dir contents
    files = os.listdir(data_pth)

    masst = list(filter(lambda name: "masst" in name, files))

    nhalos_tot = 0

    # Get halo number
    for target in masst[:]:
        print("Opening %s ... " % target)
        with open(os.path.join(data_pth, target), "rb") as ff:
            nhalos_tot += np.fromfile(ff, np.int32, count=3)[
                1
            ]  # Number between 2 buoys
            print("%s halos ... \n" % nhalos_tot)
        ff.close()

    # idx mass x y z file_nb
    halo_data = np.zeros((nhalos_tot, 6))

    # Get data
    line = 0
    loc_line = 0
    for target in masst[:]:
        print("Opening %s ... " % target)
        with open(os.path.join(data_pth, target), "rb") as ff:
            nhalos = np.fromfile(ff, np.int32, count=3)[1]  # Number between 2 buoys
            halo_data[line : line + nhalos, -1] = float(target[-4:]) * np.ones(
                (1, nhalos)
            )
            for loc_line in range(nhalos):
                buoy = np.fromfile(ff, np.int32, count=1)[0]
                idx = np.fromfile(ff, np.int64, count=1)[0]
                mass = np.fromfile(ff, np.int32, count=1)[0]
                x, y, z = np.fromfile(ff, np.float32, count=3)
                buoy = np.fromfile(ff, np.int32, count=1)[0]

                halo_data[line, :-1] = np.asarray([idx, mass, x, y, z])
                # print(halo_data[line,:])
                line += 1

        ff.close()

    masst_nbs = np.asarray([int(target.split("_")[-1]) for target in masst])

    return (halo_data, masst_nbs)


def o_luke_fof(fof_path, output_str):
    halos = {}

    with h5py.File(os.path.join(fof_path, output_str, "haloes_masst.h5"), "r") as src:
        keys = list(src["Data"].keys())
        print(keys)
        for key in keys:
            if key != "file_number":
                halos[key] = src["Data"][key][()]
            else:
                halo_fnbs = src["Data"][key][()]

    halos = np.asarray([halos[k] for k in halos.keys()]).T

    return halos, halo_fnbs


def o_mp_fof(data_path, Mp):
    dt = np.dtype(
        [
            ("struct", "i4"),
            ("idx", "i4"),
            ("idd", "i4"),
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("mass", "f8"),
        ]
    )
    with open(data_path, "rb") as src:
        data = np.fromfile(src, dt)

    # mass doesn't need to be doubles...
    new_dt = np.dtype(
        [
            ("struct", "i4"),
            ("idx", "i4"),
            ("idd", "i4"),
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("mass", "f4"),
        ]
    )
    data["mass"] = data["mass"] / Mp

    data = data.astype(new_dt)

    return_keys = ["idx", "mass", "x", "y", "z"]

    return (np.asarray([data[k] for k in return_keys]).T, data["struct"])
