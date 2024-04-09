import numpy as np
import os
from mpi4py import MPI
import h5py


def sum_arrays_to_rank0(comm, array, op=MPI.SUM):
    recvbuf = None
    if comm.rank == 0:
        recvbuf = np.empty_like(array)

    comm.Reduce(array, recvbuf, op)

    return recvbuf


def scatter_arrays_rank0(comm, array, dtype=np.int64, debug=False):
    # get size of final object

    if comm.rank == 0:
        # sendbuf = np.empty(comm.Get_size())

        sendbuf = array

        # count: the size of each sub-task
        ave, res = divmod(sendbuf.size, comm.size)
        count = [ave + 1 if p < res else ave for p in range(comm.size)]
        count = np.array(count)

        # displacement: the starting index of each sub-task
        displ = [sum(count[:p]) for p in range(comm.size)]
        displ = np.array(displ)

    else:
        sendbuf = None
        # initialize count on worker processes
        count = np.zeros(comm.size, dtype=int)
        displ = None

    # broadcast count
    comm.Bcast(count, root=0)

    # initialize recvbuf on all processes
    recvbuf = np.zeros(count[comm.rank], dtype=dtype)

    comm.Scatterv([sendbuf, count, displ, dtype], recvbuf, root=0)

    return recvbuf


def merge_arrays_rank0(comm, array, dtype=np.int64, debug=False):
    # get size of final object
    recvbuf = None
    if comm.rank == 0:
        recvbuf = np.empty(comm.Get_size())

    sendcounts = np.array(comm.gather(len(array), 0))

    if comm.rank == 0:
        if debug:
            print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
        recvbuf = np.empty(sum(sendcounts), dtype=int)
    else:
        recvbuf = None

    tot_size = np.sum(sendcounts)
    # allocate buffet on rank 0
    if comm.rank == 0:
        recvbuf = np.empty(tot_size, dtype=dtype)
        # print(tot_size)

    # gather
    comm.Gatherv(sendbuf=array, recvbuf=(recvbuf, sendcounts), root=0)

    return recvbuf


def merge_arrays_rank0_Ndd_flatten_rebuild(comm, array, dtype=np.int64, debug=False):
    """
    first dimension of multi dimensional array should be the one merged accross ranks
    ALL OTHERS ARE EXPETED TO REMAIN THE SAME
    """

    ndims = len(array.shape)
    shape = array.shape

    # print(shape)

    out = merge_arrays_rank0(comm, np.ravel(array), dtype=dtype, debug=debug)

    if comm.rank == 0:
        # print(out, tgt_shape)
        tgt_shape = (-1,) + shape[1:]
        # print(out, tgt_shape, shape)
        return out.reshape(tgt_shape)
    else:
        return None


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

    # print(path, files)

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
