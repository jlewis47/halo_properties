from tkinter import N
import numpy as np
from halo_properties.files.read_fullbox_big import *

# from read_radgpu_big import *
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

"""
Functtions for dealing with pertionic boundaries
"""


def gen_pos_vects(ldx):

    ran = [-ldx, 0, ldx]
    pos_vects = np.asarray([[i, j, k] for k in ran for j in ran for i in ran])

    return pos_vects


def get_27(pos1, pos2, pos_vects):
    """
    Returns all distances including pos_vects reflecttions
    check that pos_vects and pos are same units !!!!
    """

    # return((np.linalg.norm(pos1-(np.asarray(pos_vects)+np.asarray(pos2)[:,np.newaxis]),axis=2,ord=2)))
    return np.linalg.norm(
        pos1 - (np.asarray(pos_vects) + np.asarray(pos2)), axis=1, ord=2
    )


def get_mult_27(pos1, pos2, pos_vects):
    """
    Returns all distances including pos_vects reflecttions
    check that pos_vects and pos are same units !!!!
    """

    return np.linalg.norm(
        pos1 - (np.asarray(pos_vects) + np.asarray(pos2)[:, np.newaxis]), axis=2, ord=2
    )


# def get_27_tree(pos,tree,dist,pos_vects):
#     """
#     Return all within dist of pos, accounting for edge repetittions listed in pos_vects
#     check that pos_vects and pos are same units !!!!
#     """
#     dists=get_27(ctr,pos,pos_vects) #allows to pick the reflecttions that give the closest posittions
#     whs=tree.query_ball_point(pos+np.asarray(pos_vects)[np.argmin(dists,axis=1),dist)
#     return(whs.tolist())


def get_27_tree(pos, tree, dist, pos_vects):
    """
    Return all within dist of pos, accounting for edge repetittions listed in pos_vects
    """
    whs = np.array([])
    for pos_vect in pos_vects:
        whs = np.untion1d(whs, tree.query_ball_point(pos + pos_vect, dist))

    return whs.tolist()


# def get_27_box(pos,ctr,dist,pos_vects):
#     """
#     Return all within dist of pos, accounting for edge repetittions listed in pos_vects
#     check that pos_vects and pos are same units !!!!
#     """
#     dists=get_mult_27(ctr,pos,pos_vects)
#     whs=catch_box(pos+np.asarray(pos_vects)[np.argmin(dists,axis=1)],ctr,dist)

#     return(whs)


def get_27_tree_nearest(pos, tree, pos_vects):
    """
    Return all within dist of pos, accounting for edge repetittions listed in pos_vects
    check that pos_vects and pos are same units !!!!
    """
    whs_dists, whs_inds = tree.query(pos + pos_vects, 1, p=2)

    return whs_inds[np.argmin(whs_dists)]


def get_ctr(poss, pos_vects, weights=None):

    ctr = np.average(poss, weights=weights, axis=0)

    all_dists = get_27(ctr, poss, pos_vects)

    min_arg = np.argmin(all_dists, axis=1)
    ctr = np.average(poss + np.asarray(pos_vects)[min_arg], weights=weights, axis=0)

    return ctr


def get_ctr_mult(poss, pos_vects, weights=None):

    ctr = np.average(poss, weights=weights, axis=0)

    all_dists = get_mult_27(ctr, poss, pos_vects)

    min_arg = np.argmin(all_dists, axis=1)
    ctr = np.average(poss + np.asarray(pos_vects)[min_arg], weights=weights, axis=0)

    return ctr


def get_ctr_mult_cheap(halo_ctr, poss, ldx, weights):

    diff = halo_ctr - poss
    sign = np.sign(diff)

    new_coords = np.copy(poss)
    pb_coords = np.abs(diff) > 0.5 * ldx
    new_coords[pb_coords] += sign[pb_coords] * ldx

    new_ctr = np.average(new_coords, axis=0, weights=weights)

    return new_ctr


def get_neighbour_cubes(subcube_nb, n_subcubes):

    nrow = int(
        round(n_subcubes ** (1.0 / 3))
    )  # number of subcubes in a row for each cartesian directtion
    nsqu = int(
        round(n_subcubes ** (2.0 / 3))
    )  # number of subcubes in a square (row*row)

    # print(nrow, nsqu)

    vect = np.arange(0, nrow)
    box_of_nbs = (
        vect + vect[:, np.newaxis] * nrow + vect[:, np.newaxis, np.newaxis] * nsqu
    )

    reflected_box = np.tile(box_of_nbs, (3, 3, 3))

    x_nb, y_nb, z_nb = np.unravel_index(subcube_nb, (nrow, nrow, nrow))

    # print(z_nb,y_nb,x_nb)

    vect_rel = np.arange(-1, 2)

    z_coords = z_nb + vect_rel
    y_coords = y_nb + vect_rel
    x_coords = x_nb + vect_rel

    # print(z_coords,y_coords,x_coords)

    x_get, y_get, z_get = np.meshgrid(z_coords, y_coords, x_coords)

    # print(x_get,y_get,z_get)

    nbs_to_get = reflected_box[z_get, y_get, x_get]

    # print(subcube_nb, nbs_to_get)

    return nbs_to_get


# def get_overstep_RT_cubed(subcube_nb,data_path_rad,OOB,n_subcubes=512,size=512,overstep=3,sort=2):

#     if sort==2:
#         box=np.zeros((3,size*overstep,size*overstep,size*overstep),dtype=np.float32)
#     elif sort==1:
#         box=np.zeros((size*overstep,size*overstep,size*overstep),dtype=np.float32)

#     nbs_to_get=get_neighbour_cubes(subcube_nb,n_subcubes)

#     delta=int((overstep-1)*0.5*size) #size of overstep in cells
#     #so for ex on xvector we have 0:delta then delta:size then size:delta from
#     #3 different subcubes where the central one is subcube_nb

#     under,over=OOB

#     #so can use with only one entry in OOB
#     if np.shape(np.shape(under))!=(2,):
#         under=np.array([under])
#         over=np.array([over])


#     zbnds=under[:,0],np.ones(len(under))==1,over[:,0]
#     ybnds=under[:,1],np.ones(len(under))==1,over[:,1]
#     xbnds=under[:,2],np.ones(len(under))==1,over[:,2]


#     #print(zbnds,ybnds,xbnds)

#     for ix,x in enumerate(nbs_to_get.swapaxes(0,1)):
#         for iy,y in enumerate(x):
#             for iz,z in enumerate(y):
#     #so can use with only one entry in OOB

#                 #print(xbnds[ix],ybnds[iy],zbnds[iz])
#                 #print(np.all([xbnds[ix],ybnds[iy],zbnds[iz]],axis=0))

#                 if np.any(np.all([xbnds[ix],ybnds[iy],zbnds[iz]],axis=0)):


#                     if ix==0:
#                         xlow,xhigh=0,delta #for big cube of sides size+2*delta
#                         load_xlow,load_xhigh=size-delta,size #for cube just opened
#                     elif ix==1:
#                         xlow,xhigh=delta,size+delta
#                         load_xlow,load_xhigh=0,size #for cube just opened
#                     else:
#                         xlow,xhigh=size+delta,size+2*delta
#                         load_xlow,load_xhigh=0,delta #for cube just opened


#                     if iy==0:
#                         ylow,yhigh=0,delta #for big cube of sides size+2*delta
#                         load_ylow,load_yhigh=size-delta,size #for cube just opened
#                     elif iy==1:
#                         ylow,yhigh=delta,size+delta
#                         load_ylow,load_yhigh=0,size #for cube just opened
#                     else:
#                         ylow,yhigh=size+delta,size+2*delta
#                         load_ylow,load_yhigh=0,delta #for cube just opened

#                     if iz==0:
#                         zlow,zhigh=0,delta #for big cube of sides size+2*delta
#                         load_zlow,load_zhigh=size-delta,size #for cube just opened
#                     elif iz==1:
#                         zlow,zhigh=delta,size+delta
#                         load_zlow,load_zhigh=0,size #for cube just opened
#                     else:
#                         zlow,zhigh=size+delta,size+2*delta
#                         load_zlow,load_zhigh=0,delta #for cube just opened


#                     if sort==2:

#                             print('RT loaded %i'%z)
#                             try:

#                                 #box[:,ix*size:(ix+1)*size,iy*size:(iy+1)*size,iz*size:(iz+1)*size]=o_rad_cube_big(data_path_rad,2,z)
#                                 box[:,xlow:xhigh,ylow:yhigh,zlow:zhigh]=o_rad_cube_big(data_path_rad,2,z)[load_zlow:load_zhigh,load_ylow:load_yhigh,load_xlow:load_xhigh]


#                                 #print([xlow,xhigh,ylow,yhigh,zlow,zhigh],[load_zlow,load_zhigh,load_ylow,load_yhigh,load_xlow,load_xhigh])
#                                 #print([ix*size,(ix+1)*size,iy*size,(iy+1)*size,iz*size,(iz+1)*size])
#                             except IndexError:
#                                 box[:,xlow:xhigh,ylow:yhigh,zlow:zhigh]=-1
#                                 continue


#                     elif sort==1:

#                             print('RT loaded %i'%z)
#                             try:
#                                box[xlow:xhigh,ylow:yhigh,zlow:zhigh]=o_rad_cube_big(data_path_rad,2,z)[load_xlow:load_xhigh,load_ylow:load_yhigh,load_zlow:load_zhigh]
#                             except IndexError:
#                                 box[xlow:xhigh,ylow:yhigh,zlow:zhigh]=-1
#                                 continue

#     return(box)


def get_overstep_hydro_cubed(
    box,
    subcube_nb,
    data_path,
    name,
    OOB,
    n_subcubes=512,
    size=512,
    overstep=3,
    debug=False,
):

    # we get this from the function call so python handles memory stuff a big
    # box=np.zeros((size*overstep,size*overstep,size*overstep),dtype=np.float32)

    nbs_to_get = get_neighbour_cubes(subcube_nb, n_subcubes)

    # nrows=round(n_subcubes**(1./3))
    # ncols=round(n_subcubes**(2./3))

    delta = int((overstep - 1) * 0.5 * size)  # size of overstep in cells
    # so for ex on xvector we have 0:delta then delta:size then size:delta from
    # 3 different subcubes where the central one is subcube_nb

    under, over = OOB

    # so can use with only one entry in OOB
    if np.shape(np.shape(under)) != (2,):
        under = np.array([under])
        over = np.array([over])

    # print(nbs_to_get.swapaxes(0,1))

    zbnds = under[:, 0], np.ones(len(under)) == 1, over[:, 0]
    ybnds = under[:, 1], np.ones(len(under)) == 1, over[:, 1]
    xbnds = under[:, 2], np.ones(len(under)) == 1, over[:, 2]

    # first we find the subcubes we need by checking all xbnds,ybnds,zbnds
    subs_required = []

    for ix, x in enumerate(nbs_to_get.swapaxes(0, 1)):
        for iy, y in enumerate(x):
            for iz, z in enumerate(y):

                # print(xbnds[ix],ybnds[iy],zbnds[iz])
                # print(np.all([xbnds[ix],ybnds[iy],zbnds[iz]],axis=0))

                if np.any(np.all([xbnds[ix], ybnds[iy], zbnds[iz]], axis=0)):
                    subs_required.append(z)

    # print(subs_required)
    # for every unique subcube number that we require we load the subcube
    # then we iterate over central and surrounding data cube using the currently loaded subcube when necessary

    nbs = 0

    for n_subcube in np.unique(subs_required):

        data_name = os.path.join(data_path, "%s_%05i" % (name, n_subcube))

        if debug:
            print("loaded %s_%i" % (name, n_subcube))
        try:
            cur_box = o_data(data_name)
            # cur_box=np.ones((size,size,size))*n_subcube
        except IndexError:
            print("Missing box assuming this is known ... Filling with 0s")
            cur_box = np.zeros((size, size, size))
            continue

        for ix, x in enumerate(nbs_to_get.swapaxes(0, 1)):
            for iy, y in enumerate(x):
                for iz, z in enumerate(y):
                    # so can use with only one entry in OOB

                    if z != n_subcube:
                        continue

                    # print(xbnds[ix],ybnds[iy],zbnds[iz])
                    # print(np.all([xbnds[ix],ybnds[iy],zbnds[iz]],axis=0))

                    if np.any(np.all([xbnds[ix], ybnds[iy], zbnds[iz]], axis=0)):

                        if ix == 0:
                            xlow, xhigh = 0, delta  # for big cube of sides size+2*delta
                            load_xlow, load_xhigh = (
                                size - delta,
                                size,
                            )  # for cube just opened
                        elif ix == 1:
                            xlow, xhigh = delta, size + delta
                            load_xlow, load_xhigh = 0, size  # for cube just opened
                        else:
                            xlow, xhigh = size + delta, size + 2 * delta
                            load_xlow, load_xhigh = 0, delta  # for cube just opened

                        if iy == 0:
                            ylow, yhigh = 0, delta  # for big cube of sides size+2*delta
                            load_ylow, load_yhigh = (
                                size - delta,
                                size,
                            )  # for cube just opened
                        elif iy == 1:
                            ylow, yhigh = delta, size + delta
                            load_ylow, load_yhigh = 0, size  # for cube just opened
                        else:
                            ylow, yhigh = size + delta, size + 2 * delta
                            load_ylow, load_yhigh = 0, delta  # for cube just opened

                        if iz == 0:
                            zlow, zhigh = 0, delta  # for big cube of sides size+2*delta
                            load_zlow, load_zhigh = (
                                size - delta,
                                size,
                            )  # for cube just opened
                        elif iz == 1:
                            zlow, zhigh = delta, size + delta
                            load_zlow, load_zhigh = 0, size  # for cube just opened
                        else:
                            zlow, zhigh = size + delta, size + 2 * delta
                            load_zlow, load_zhigh = 0, delta  # for cube just opened

                        # print(xlow,xhigh,ylow,yhigh,zlow,zhigh)
                        # print(load_xlow,load_xhigh,load_ylow,load_yhigh,load_zlow,load_zhigh)

                        # print(z,ix,iy,iz)

                        # print(ix,iy,iz, n_subcube, nbs)

                        box[xlow:xhigh, ylow:yhigh, zlow:zhigh] = cur_box[
                            load_xlow:load_xhigh,
                            load_ylow:load_yhigh,
                            load_zlow:load_zhigh,
                        ]

                        nbs += 1

    # fig=plt.figure()
    # ax=fig.add_subplot(111)

    # img=ax.imshow(np.log10(box[510,:,:]).T,origin='lower',vmin=-3,vmax=10)
    # ax.set_xlabel('y')
    # ax.set_ylabel('x')
    # plt.colorbar(img)
    # fig.savefig('test_%i_xy'%subcube_nb)

    # img=ax.imshow(np.log10(box[:,:,510]).T,origin='lower',vmin=-3,vmax=10)
    # ax.set_xlabel('z')
    # ax.set_ylabel('y')

    # fig.savefig('test_%i_yz'%subcube_nb)

    # img=ax.imshow(np.log10(box[:,510,:]).T,origin='lower',vmin=-3,vmax=10)
    # ax.set_xlabel('z')
    # ax.set_ylabel('x')

    # fig.savefig('test_%i_xz'%subcube_nb)

    return box


def new_wrap_box(subnb, nb_subcubes):

    n_per_row = int(np.round(np.cbrt(nb_subcubes)))
    n_per_plane = int(np.round(np.cbrt(nb_subcubes) ** 2.0))

    # 27 surrounding boxes
    dims = 3
    sub_ids = np.zeros((dims, dims, dims), dtype=np.int16)

    for ix in range(dims):
        for iy in range(dims):
            for iz in range(dims):

                sub_ids[iz, iy, ix] = new_wrap_single(subnb, nb_subcubes, ix, iy, iz)

    return sub_ids


def new_wrap_single(subnb, nb_subcubes, ix, iy, iz):
    def prep_mod(coord, ind, size):  # handle edges

        mod = (ind - 1) + coord
        if mod < 0:
            mod += size
        if mod >= size:
            mod -= size

        return mod

    n_per_row = int(np.round(np.cbrt(nb_subcubes)))

    x, y, z = np.unravel_index(subnb, (n_per_row, n_per_row, n_per_row))

    modx = prep_mod(x, ix, n_per_row)
    mody = prep_mod(y, iy, n_per_row)
    modz = prep_mod(z, iz, n_per_row)

    # print(subnb,modz, mody, modx)

    nb = np.ravel_multi_index((modx, mody, modz), (n_per_row, n_per_row, n_per_row))

    return nb


def new_get_overstep_hydro_cubed(
    box,
    subcube_nb,
    data_path,
    name,
    OOB,
    n_subcubes=512,
    size=512,
    overstep=3,
    debug=False,
):

    delta = int((overstep - 1) * 0.5 * size)  # size of overstep in cells
    # so for ex on xvector we have 0:delta then delta:size then size:delta from
    # 3 different subcubes where the central one is subcube_nb

    under, over = OOB

    # so can use with only one entry in OOB
    if np.shape(np.shape(under)) != (2,):
        under = np.array([under])
        over = np.array([over])

    nbs = 0

    zbnds = under[:, 0], np.ones(len(under)) == 1, over[:, 0]
    ybnds = under[:, 1], np.ones(len(under)) == 1, over[:, 1]
    xbnds = under[:, 2], np.ones(len(under)) == 1, over[:, 2]

    dims = 3

    for ix in range(dims):
        for iy in range(dims):
            for iz in range(dims):

                # print(xbnds[ix],ybnds[iy],zbnds[iz])
                # print(np.all([xbnds[ix],ybnds[iy],zbnds[iz]],axis=0))

                if np.any(np.all([xbnds[ix], ybnds[iy], zbnds[iz]], axis=0)):

                    nb_to_get = new_wrap_single(subcube_nb, n_subcubes, iz, iy, ix)

                    # print(nb_to_get, subcube_nb, n_subcubes, iz, iy, ix)

                    data_name = os.path.join(data_path, "%s_%05i" % (name, nb_to_get))

                    if debug:
                        print("loaded %s_%i" % (name, nb_to_get))

                    if ix == 0:
                        xlow, xhigh = 0, delta  # for big cube of sides size+2*delta
                        load_xlow, load_xhigh = (
                            size - delta,
                            size,
                        )  # for cube just opened
                    elif ix == 1:
                        xlow, xhigh = delta, size + delta
                        load_xlow, load_xhigh = 0, size  # for cube just opened
                    else:
                        xlow, xhigh = size + delta, size + 2 * delta
                        load_xlow, load_xhigh = 0, delta  # for cube just opened

                    if iy == 0:
                        ylow, yhigh = 0, delta  # for big cube of sides size+2*delta
                        load_ylow, load_yhigh = (
                            size - delta,
                            size,
                        )  # for cube just opened
                    elif iy == 1:
                        ylow, yhigh = delta, size + delta
                        load_ylow, load_yhigh = 0, size  # for cube just opened
                    else:
                        ylow, yhigh = size + delta, size + 2 * delta
                        load_ylow, load_yhigh = 0, delta  # for cube just opened

                    if iz == 0:
                        zlow, zhigh = 0, delta  # for big cube of sides size+2*delta
                        load_zlow, load_zhigh = (
                            size - delta,
                            size,
                        )  # for cube just opened
                    elif iz == 1:
                        zlow, zhigh = delta, size + delta
                        load_zlow, load_zhigh = 0, size  # for cube just opened
                    else:
                        zlow, zhigh = size + delta, size + 2 * delta
                        load_zlow, load_zhigh = 0, delta  # for cube just opened

                    # have tested a few loading techniques
                    # fastest is memmap into pre-allocated empty numpy array of correct size
                    # total timeits for np.memmap, np.fromfile, hdf5 give 23.5s, 36.1s, 27s

                    # try:
                    # cur_box = o_data(data_name)
                    # cur_box = o_data_memmap(data_name)

                    # box[xlow:xhigh, ylow:yhigh, zlow:zhigh] = o_data_hdf5(
                    #     data_name + ".hdf5",
                    #     name,
                    #     (
                    #         (load_xlow, load_xhigh),
                    #         (load_ylow, load_yhigh),
                    #         (load_zlow, load_zhigh),
                    #     ),
                    # )
                    # box[xlow:xhigh, ylow:yhigh, zlow:zhigh] = o_data_memmap(
                    #     data_name
                    # )[
                    #     load_xlow:load_xhigh,
                    #     load_ylow:load_yhigh,
                    #     load_zlow:load_zhigh,
                    # ]
                    try:
                        # cur_box = o_data(data_name)
                        #     # cur_box = o_data_memmap(data_name)
                        box[xlow:xhigh, ylow:yhigh, zlow:zhigh] = o_data_memmap(
                            data_name,
                            (
                                (load_xlow, load_xhigh),
                                (load_ylow, load_yhigh),
                                (load_zlow, load_zhigh),
                            ),
                        )

                    # cur_box=np.ones((size,size,size))*n_subcube
                    except IndexError:
                        print("Missing box assuming this is known ... Filling with 0s")
                        cur_box = np.zeros((size, size, size))
                        continue

                    # nbs+=1

    # return box
