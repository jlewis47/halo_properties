import numpy as np
import os
from mpi4py import MPI
import h5py


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


def sum_arrays_to_rank0(comm, array, op=MPI.SUM):
    recvbuf = None
    if comm.rank == 0:
        recvbuf = np.empty_like(array)

    comm.Reduce(array, recvbuf, op)

    return recvbuf


def merge_arrays_rank0(comm, array, dtype=np.int64):
    # get size of final object
    recvbuf = None
    if comm.rank == 0:
        recvbuf = np.empty(comm.Get_size())

    sendcounts = np.array(comm.gather(len(array), 0))

    if comm.rank == 0:
        print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
        recvbuf = np.empty(sum(sendcounts), dtype=int)
    else:
        recvbuf = None

    tot_size = np.sum(sendcounts)
    # allocate buffet on rank 0
    if comm.rank == 0:
        recvbuf = np.empty(tot_size, dtype=dtype)

    # gather
    comm.Gatherv(sendbuf=array, recvbuf=(recvbuf, sendcounts), root=0)

    return recvbuf


def divide_task(nb_files, Nproc, rank):
    f_per_proc = int(np.floor(float(nb_files - 1) / float(Nproc)))
    fmin = rank * f_per_proc
    fmax = (rank + 1) * f_per_proc

    tot = f_per_proc * Nproc
    missing = nb_files - tot

    if missing > 0:
        if rank < missing:
            fmin += rank
            fmax += rank + 1
        else:
            fmin += missing
            fmax += missing

    return (fmin, fmax, f_per_proc)


def divide_task_space(ldx, Nproc, rank):
    NperSide = (Nproc) ** (1.0 / 3.0)
    lbox = int(np.round(ldx / NperSide))

    # print(NperSide, lbox)

    # print(rank, list(map(lambda x : int(np.round(x)),[NperSide, NperSide, NperSide])))

    x, y, z = np.unravel_index(
        rank, list(map(lambda x: int(np.round(x)), [NperSide, NperSide, NperSide]))
    )

    return (x, y, z, lbox)


def gather_h5py_files(path, keys=None, rank=0, Nrank=1):
    files = [os.path.join(path, f) for f in os.listdir(path) if "halo_stats" in f]
    fnbs = [int(f.split("/")[-1].split("_")[-1]) for f in files]

    fmin, fmax, f_per_proc = divide_task(len(files), Nrank, rank)

    files = files[fmin:fmax]

    data_len = 0

    types = []
    dims = []

    with h5py.File(files[0], "r") as src:
        if keys == None:
            keys = list(map(str, src.keys()))
            print("Available keys are %s" % keys)

        for k in keys:
            types.append(src[k].dtype.name)
            dim = np.shape(src[k])

            if len(dim) > 1:
                dim = dim[1]
            else:
                dim = 1

            dims.append(dim)
        data_len += src[k].len()

    for f in files[1:]:
        try:
            with h5py.File(f, "r") as src:
                data_len += src[k].len()
        except (OSError, KeyError) as e:
            print(e)
            print(f"Error for file {f:s}")
            continue

    dtype = [(k, typ, dim) for typ, k, dim in zip(types, keys, dims)]
    datas = np.empty(data_len, dtype=dtype)

    tot_l = 0
    for f in files:
        try:
            with h5py.File(f, "r") as src:
                for ik, k in enumerate(keys):
                    loc_data = src[k]
                    loc_l = len(loc_data)
                    datas[k][tot_l : tot_l + loc_l] = loc_data
                tot_l += loc_l
        except (OSError, KeyError):
            continue

    return datas


def merge_hdf5_files(path, keys=None):
    datas = gather_h5py_files(path, keys)

    with h5py.File(os.path.join(path, "halo_stats_merged"), "w") as dest:
        for key in datas.dtype:
            dat = datas[key]
            dest.create_dataset(key, dat, dtype=dat.dtype, compresstion="lzf")
