import numpy as np
from os import listdir, path as p
from ..utils import utils
from .wrap_boxes import new_wrap_single
from scipy.spatial import cKDTree


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

        datas = np.fromfile(src, dtype=dt)
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


def read_stars_cutout(path, ctr, size, ldx=8192):
    star_files = listdir(path)
    star_file_nbs = [int(f.split("_")[-1]) for f in star_files]
    nst_files = len(star_files)
    nst_files_pside = int(np.cbrt(nst_files))
    subsize = ldx // nst_files_pside

    # file_xs, file_ys, file_zs = np.unravel_index(
    #     star_file_nbs, (nst_files_pside, nst_files_pside, nst_files_pside)
    # )

    # file_l = 1.0 / nst_files_pside

    ctr = np.asarray(ctr)
    ctr_subs = np.int32(ctr // (subsize))
    ctr_in_sub = np.int32(ctr % (subsize))
    n_subcubes = int(ldx**3 // subsize**3)
    n_subs_per_side = int(np.cbrt(n_subcubes))

    # find central subsize sized box to load
    ctr_subnb = np.ravel_multi_index(
        ctr_subs, (n_subs_per_side, n_subs_per_side, n_subs_per_side)
    )

    lo_lims = ctr_in_sub - size // 2
    hi_lims = ctr_in_sub + size // 2

    oversteps = np.array([lo_lims - 0, hi_lims - (subsize - 1)])

    oversteps_bool = [oversteps[0] < 0, oversteps[1] > 0]

    assert not np.any(
        np.all(oversteps_bool, axis=0)
    ), "ctr, size, and subsize mean that overstepping in at least two diretions... not supported use other functions"

    # print(ctr_subnb)
    # print(ctr_in_sub)
    # print(lo_lims, hi_lims)
    # print(oversteps)
    # print(oversteps_bool)

    zbnds = oversteps_bool[0][0], np.full(1, True), oversteps_bool[1][0]
    ybnds = oversteps_bool[0][1], np.full(1, True), oversteps_bool[1][1]
    xbnds = oversteps_bool[0][2], np.full(1, True), oversteps_bool[1][2]

    # print(len(fields), size)

    dims = 3

    out_stars = []

    for ix in range(dims):
        for iy in range(dims):
            for iz in range(dims):
                # print(xbnds[ix], ybnds[iy], zbnds[iz])
                # print(np.all([xbnds[ix], ybnds[iy], zbnds[iz]], axis=0))

                if np.any(np.all([xbnds[ix], ybnds[iy], zbnds[iz]], axis=0)):
                    nb_to_get = new_wrap_single(ctr_subnb, n_subcubes, iz, iy, ix)

                    stars = read_star_file(p.join(path, "stars_%i" % nb_to_get))[1]

                    x, y, z = stars["x"], stars["y"], stars["z"]

                    star_tree = cKDTree(
                        np.transpose([x, y, z]),
                        boxsize=1.00001,
                    )

                    cond = star_tree.query_ball_point(ctr / ldx, size / ldx)

                    out_stars.append([stars[key][cond] for key in stars.dtype.names])

    return np.concatenate(out_stars, axis=1)
