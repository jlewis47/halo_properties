import numpy as np
from os import listdir, path as p
from ..utils import utils


def get_nb_lines(in_path, dt_line=None):

    if dt_line == None:
        dt_line = np.dtype(
            [
                ("buf1", "i4"),
                ("id", "i4"),
                ("mass", "f8"),
                ("x", "f8"),
                ("y", "f8"),
                ("z", "f8"),
                ("age", "f8"),
                ("Z/0.02", "f8"),
                ("ll", "i4"),
                ("tag", "i1"),
                ("family", "i1"),
                ("buf2", "i4"),
            ]
        )

    byte_size = p.getsize(in_path)
    bytes_per_line = dt_line.itemsize

    nb_lines = byte_size - (4 + 8 + 4)  # remove universe age and buffers
    nb_lines = byte_size / bytes_per_line

    assert int(nb_lines) == int(
        nb_lines // 1
    ), "found a non integer number of lines %i in file %s" % (nb_lines, in_path)

    return int(nb_lines)


def read_star_file(in_path):
    """
    masses are in units of solar mass
    ages are in Myr
    x,y,z are in box length units (>0, <1)
    metallicities are in solar units
    """
    dt = np.dtype(
        [
            ("buf1", "i4"),
            ("id", "i4"),
            ("mass", "f8"),
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("age", "f8"),
            ("Z/0.02", "f8"),
            ("ll", "i4"),
            ("tag", "i1"),
            ("family", "i1"),
            ("buf2", "i4"),
        ]
    )

    nb_lines = get_nb_lines(in_path, dt)

    datas = np.empty(nb_lines, dtype=dt)
    with open(in_path, "rb") as src:

        buf = np.fromfile(src, dtype=np.int32, count=1)
        time_myr = np.fromfile(src, dtype=np.float64, count=1)
        buf = np.fromfile(src, dtype=np.int32, count=1)

        datas=np.fromfile(src, dtype=dt)
        # datas.fromfile(src, dtype=dt)

    return (
        time_myr,
        datas[["id", "mass", "x", "y", "z", "age", "Z/0.02", "family"]],
        np.arange(nb_lines, dtype=np.int64),
    )


def read_all_star_files(tgt_path):

    dt = np.dtype(
        [
            ("id", "i4"),
            ("mass", "f8"),
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("age", "f8"),
            ("Z/0.02", "f8"),
            ("family", "i1"),
        ]
    )

    targets = [p.join(tgt_path, f) for f in listdir(tgt_path) if "star" in f]
    fnbs = np.sort([int(target.split("_")[-1]) for target in targets])

    size = 0
    for target in targets:

        size += get_nb_lines(target)

    datas = np.zeros((size), dtype=dt)
    joe_ids = np.zeros((size), dtype=np.int32)

    size = 0
    for fnb, target in zip(fnbs, targets[:]):

        time_myr, loc_data, loc_nbs = read_star_file(target)

        l = len(loc_data["mass"])

        datas[:][size : size + l] = loc_data[:][:]
        joe_ids[size : size + l] = pair(fnb, loc_nbs)

        size += l

    return (time_myr, datas, joe_ids)


def read_rank_star_files(tgt_path, rank, Nproc):

    dt = np.dtype(
        [
            ("id", "i4"),
            ("mass", "f8"),
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("age", "f8"),
            ("Z/0.02", "f8"),
            ("family", "i1"),
        ]
    )

    targets = np.asarray(
        [p.join(tgt_path, f) for f in listdir(tgt_path) if "star" in f]
    )
    order = np.argsort([int(target.split("_")[-1]) for target in targets])
    targets = targets[order]

    nb_files = len(targets)

    if rank == 0:
        print(
            "Found %i stellar files, distributing association accross %i ranks"
            % (nb_files, Nproc)
        )

    nmin, nmax, nperProc = utils.divide_task(nb_files, Nproc, rank)

    targets = targets[nmin:nmax]
    fnbs = [int(target.split("_")[-1]) for target in targets]

    print("rank %i gets files (%i,%i)" % (rank, nmin, nmax))
    print(fnbs)

    size = 0
    for target in targets:

        size += get_nb_lines(target)

    datas = np.zeros((size), dtype=dt)
    joe_ids = np.zeros((size), dtype=np.int64)

    size = 0
    for fnb, target in zip(fnbs, targets[:]):

        time_myr, loc_data, loc_nbs = read_star_file(target)

        l = len(loc_data["mass"])

        datas[:][size : size + l] = loc_data[:][:]
        joe_ids[size : size + l] = pair(fnb, loc_nbs)

        size += l

    return (time_myr, datas, joe_ids)


def pair(fnb, lnb):
    return np.int64(lnb + fnb * 1e10)


def unpair(pairs):
    fnbs = pairs // 1e10
    lnbs = pairs % 1e10

    return (np.int64(fnbs), np.int64(lnbs))

def read_specific_stars(path, pairs, keys=None, formatted=True):

    """
    path to binary star files
    pairs are the IDs containing star file and line nb informattion
    formatted=True => return a structured numpy array, else return a "normal" numpy array
    """

    possible_keys = ["id", "mass", "x", "y", "z", "age", "Z/0.02", "family"]
    possible_types = ["i4", "f8", "f8", "f8", "f8", "f8", "f8", "i1"]

    # convert pairs to file numbers and line numbers
    fnbs, lnbs = unpair(pairs)

    # group by file number
    fnb_sort = np.argsort(fnbs)
    fnbs = fnbs[fnb_sort]
    lnbs = lnbs[fnb_sort]

    # allocate arrays to hold stellar info ... only certain fields?
    if formatted:
        if keys == None:
            keys = possible_keys
            types = possible_types
        else:
            types = [
                typ for typ, key in zip(possible_types, possible_keys) if key in keys
            ]
            keys = [key for key in possible_keys if key in keys]

        dt = np.dtype([(key, typ) for key, typ in zip(keys, types)])
        stars = np.zeros(len(lnbs), dtype=dt)
    else:
        stars = np.zeros((len(lnbs), len(possible_keys)))

    star_counter = 0

    # find indexes where fnb changes
    # fnbs_change = np.where(np.diff(unique_fnbs) > 0)[0]
    unique_fnbs, counts_fnbs = np.unique(fnbs, return_counts=True)

    # print(len(stars), unique_fnbs.max(), counts_fnbs[np.argmax(unique_fnbs)])


    # loop over files
    # for ifnb, fnb_chng in enumerate(fnbs_change):
    for ifnb, (fnb, f_stars) in enumerate(zip(unique_fnbs, counts_fnbs)):

        time_myr, datas, joe_ids = read_star_file(p.join(path, "stars_%i" % fnb))
        # print(len(datas), len(lnbs[fnb==fnbs]), lnbs[fnb==fnbs].max())

        if formatted:
            stars[keys][star_counter : star_counter + f_stars] = datas[keys][
                lnbs[fnb == fnbs]
            ]
        else:
            stars[star_counter : star_counter + f_stars, :] = datas[keys][
                lnbs[fnb == fnbs]
            ]

        star_counter += f_stars

    return stars




