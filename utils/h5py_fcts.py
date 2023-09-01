import h5py
import os
import numpy as np


def gather_h5py_files(path, keys):
    files = [os.path.join(path, f) for f in os.listdir(path) if "halo_stats" in f]

    data_len = 0

    types = []
    cols = []
    with h5py.File(files[0], "r") as src:
        if keys == None:
            keys = src.keys()

        found_keys = np.asarray([k in src.keys() for k in keys])

        if not np.all(found_keys):
            raise Exception(
                f"key {np.asarray(keys)[found_keys==False]} not in file {files[0]}"
            )

        for k in keys:
            types.append(src[k].dtype.name)
            shape = np.shape(src[k])
            if len(shape) > 1:
                cols.append(shape[1])
            else:
                cols.append(1)

        data_len += src[k].len()

    # print(cols,types)

    for f in files[1:]:
        try:
            with h5py.File(f, "r") as src:
                data_len += src[keys[0]].len()
        except OSError:
            print(f"pb with file {f:s}... skipping")
            files.remove(f)
            continue

    dtype = [(k, typ, col) for k, typ, col in zip(keys, types, cols)]
    # print(dtype)

    datas = np.empty(data_len, dtype=dtype)

    tot_l = 0
    for f in files:
        with h5py.File(f, "r") as src:
            for ik, k in enumerate(keys):
                loc_data = src[k][()]
                loc_l = len(loc_data)
                if cols[ik] > 1:
                    datas[k][tot_l : tot_l + loc_l, :] = loc_data
                else:
                    datas[k][tot_l : tot_l + loc_l] = loc_data
            tot_l += loc_l

    return (datas[k] for k in keys)

def save_dict_to_hdf5(f, d):

    for k, v in d.items():
        if isinstance(v, dict):
            save_dict_to_hdf5(f, v)
        else:
            f.create_dataset(k, data=v)

def save_snapshot_hdr(f, snapshot_info):

    with h5py.File(f, "a") as dest:

        if not "header" in dest and not "hdr" in dest and not "Header" in dest:
            hdr = dest.create_group("header")
        else:
            if "header" in dest:
                hdr = dest["header"]
            elif "hdr" in dest:
                hdr = dest["hdr"]
            elif "Header" in dest:
                hdr = dest["Header"]

        save_dict_to_hdf5(hdr, snapshot_info)
            
