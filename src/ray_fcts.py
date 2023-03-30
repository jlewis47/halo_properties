import numpy as np
from matplotlib import patches as pat
import matplotlib.pyplot as plt
from time import sleep
# from math import cos, sin
from numba import jit,njit

def cart_2_sph(x, y, z):

    """
    Takes radians !
    """

    r = (x**2 + y**2 + z**2) ** 0.5

    phi = np.arctan2(y, x)  # radians

    the = np.arccos(z / r)  # radians

    return (r, phi, the)


def sph_2_cart(r, phi, the):

    """
    Take radians !
    """

    sin_the = np.sin(the)

    x = r * sin_the * np.cos(phi)

    y = r * sin_the * np.sin(phi)

    z = r * np.cos(the)

    return (x, y, z)

    

@njit()
def sph_2_cart_numba(r, phi, the):

    """
    Take radians !
    """

    sin_the = np.sin(the)

    x = r * sin_the * np.cos(phi)

    y = r * sin_the * np.sin(phi)

    z = r * np.cos(the)

    return (x, y, z)

@njit()
def sph_2_cart_numba_loop(r, phi, the):

    """
    Take radians !
    """

    n=len(r)
    x = np.empty_like(r)
    y = np.empty_like(r)
    z = np.empty_like(r)

    for i in range(n):


        sin_the = np.sin(the[i])

        x[i] = r[i] * sin_the * np.cos(phi[i])

        y[i] = r[i] * sin_the * np.sin(phi[i])

        z[i] = r[i] * np.cos(the[i])

    return (x, y, z)


def sum_at_r200(field, r200, Xvects, Yvects, Zvects):

    """
    sum over surfaces at Xs,Ys,Zs
    """

    size = max(np.shape(field))

    ctr = [int(0.5 * size) + 0.5] * 3

    Xs, Ys, Zs = Xvects * r200, Yvects * r200, Zvects * r200

    # used for getting data ... Need to be between 0 and size !!!
    Xs_snap, Ys_snap, Zs_snap = np.int32([Xs + ctr[0], Ys + ctr[1], Zs + ctr[2]])

    # print(size,ctr)

    # print(Xs,Xs_snap)

    sampled = field[:, Xs_snap, Ys_snap, Zs_snap]

    normal_proj = Xvects * sampled[2] + Yvects * sampled[1] + Zvects * sampled[0]

    # print(r200,size,np.shape(sampled))

    # print(normal_proj)

    # plt.figure()

    # plt.imshow((field[0,int(0.5*size),:,:]),origin='lower',extent=[0,size,0,size])
    # plt.colorbar()

    # plt.quiver(Xs+0.5*size,Ys+0.5*size,Xvects,Yvects,color='r')
    # plt.quiver(Xs+0.5*size,Ys+0.5*size,sampled[2],sampled[1],color='k')
    # plt.quiver(Xs+0.5*size,Ys+0.5*size,normal_proj*Xvects,normal_proj*Yvects,color='magenta')

    # print(np.su(mnormal_proj))

    # plt.show()

    return np.sum(normal_proj)


def sum_over_rays(field, ctr, r200, rad_res, X_primes, Y_primes, Z_primes):

    """
    compute sum over rays centred at ctr using given resoluttion
    field is a box to sum

    """

    size = np.shape(field)[0]

    ctr = np.asarray(ctr) - 0.5 * size
    delta_R = np.copy(ctr)
    Xs, Ys, Zs = [X_primes, Y_primes, Z_primes] + delta_R[:, np.newaxis, np.newaxis]

    Rs = np.linalg.norm([Xs, Ys, Zs], axis=0)

    # used for getting data ... Need to be between 0 and size !!!
    Xs_snap, Ys_snap, Zs_snap = np.int32(
        [Xs + 0.5 * size, Ys + 0.5 * size, Zs + 0.5 * size]
    )

    IB = Rs <= r200  # if points in r200 ...
    OOB = ~IB

    # print(np.shape(OOB),np.shape(sampled))
    sampled = np.zeros_like(Xs)
    sampled[IB] = field[Xs_snap[IB], Ys_snap[IB], Zs_snap[IB]]

    # sampled[Rs>r200]=0

    # if paths
    x_matches = np.zeros((np.shape(Xs)[0]), dtype=np.float32)
    y_matches = np.copy(x_matches)
    z_matches = np.copy(y_matches)

    argmin = np.argmin(np.abs(Rs[:, :] - r200), axis=1)

    # if paths
    inds = np.arange(len(x_matches))
    x_matches[inds] = Xs[inds, argmin]
    y_matches[inds] = Ys[inds, argmin]
    z_matches[inds] = Zs[inds, argmin]

    # if norms are 0 then we are on the border and the result should be 1
    # scal[np.isnan(scal)]=1

    scal = (
        (x_matches - ctr[0]) * x_matches
        + (y_matches - ctr[1]) * y_matches
        + (z_matches - ctr[2]) * z_matches
    ) / (
        (
            np.linalg.norm([x_matches, y_matches, z_matches], axis=0)
            * np.linalg.norm(
                [x_matches - ctr[0], y_matches - ctr[1], z_matches - ctr[2]], axis=0
            )
        )
    )

    scal[np.isnan(scal)] = 1

    rays = np.exp(-np.sum(sampled, axis=1) * rad_res * scal)

    return rays


def sum_over_rays_bias(
    field, ctr, r200, rad_res, X_primes, Y_primes, Z_primes, debug=False
):

    """
    compute sum over rays centred at ctr using given resoluttion
    field is a box to sum

    corrects for angular bias

    """

    size = np.shape(field)[0]

    ctr = np.asarray(ctr) - 0.5 * size
    Xs, Ys, Zs = [X_primes, Y_primes, Z_primes] + ctr[:, np.newaxis, np.newaxis]

    Rs = np.linalg.norm([Xs, Ys, Zs], axis=0)

    # used for getting data ... Need to be between 0 and size !!!
    Xs_snap, Ys_snap, Zs_snap = np.int32(
        [Xs + 0.5 * size, Ys + 0.5 * size, Zs + 0.5 * size]
    )
    # Rs_start = np.linalg.norm(
    #     [Xs_snap - ctr[0], Ys_snap - ctr[1], Zs_snap - ctr[2]], axis=0
    # )

    # print(Xs_snap,Ys_snap,Zs_snap)

    IB = Rs <= r200  # if points in r200 ...
    # OOB = ~IB

    # print(list(map(lambda x : (np.min(x),np.max(x)),[Xs_snap[IB],Ys_snap[IB],Zs_snap[IB]])))

    # print(IB, np.shape(field), np.shape(Xs))

    sampled = np.zeros_like(Xs)
    sampled[IB] = field[Zs_snap[IB], Ys_snap[IB], Xs_snap[IB]]

    # if paths
    x_matches = np.zeros((np.shape(Xs)[0]), dtype=np.float32)
    y_matches = np.copy(x_matches)
    z_matches = np.copy(y_matches)

    argmin = np.argmin(np.abs(Rs[:, :] - r200), axis=1)

    inds = np.arange(len(x_matches))
    x_matches[inds] = Xs[inds, argmin]
    y_matches[inds] = Ys[inds, argmin]
    z_matches[inds] = Zs[inds, argmin]

    # if norms are 0 then we are on the border and the result should be 1
    # scal[np.isnan(scal)]=1

    scal = (
        (x_matches - ctr[0]) * x_matches
        + (y_matches - ctr[1]) * y_matches
        + (z_matches - ctr[2]) * z_matches
    ) / (
        (
            np.linalg.norm([x_matches, y_matches, z_matches], axis=0)
            * np.linalg.norm(
                [x_matches - ctr[0], y_matches - ctr[1], z_matches - ctr[2]], axis=0
            )
        )
    )

    scal[np.isnan(scal)] = 1
    scal[np.isinf(scal)] = 1
    # np.nan_to_num(scal,nan=1,posinf=1,neginf=1)

    rays = np.exp(-np.sum(sampled, axis=1) * rad_res * scal)

    weights = (Rs[inds, argmin] ** -2) / np.sum(Rs[inds, argmin] ** -2)

    # print(np.sum(np.isinf(scal)),np.sum(np.isinf(rays)),np.sum(np.isinf(weights)),(np.sum(sampled,axis=1)*rad_res*scal)[np.isinf(rays)],(scal)[np.isinf(rays)],flush=True)

    if debug:  # r200>10:

        plt.figure(figsize=(10, 10))
        plt.subplot(111)
        plt.title("")
        plt.grid(True)
        # print(field[int(ctr[2]+0.5*size),:,:])
        # print(-np.sum(sampled,axis=1)*rad_res*scal)
        # plt.imshow(np.log10(field[int(ctr[2]+0.5*size),:,:]*rad_res),origin='lower',extent=[0,size,0,size])
        plt.imshow(
            np.log10(np.sum(field[:, :, :] * rad_res, axis=0)),
            origin="lower",
            extent=[0, size, 0, size],
        )

        plt.colorbar()

        p = pat.Circle(
            (0.5 * size, 0.5 * size),
            r200,
            color="magenta",
            fill=False,
            edgecolor="k",
            linewidth=2,
        )
        axis = plt.gca()
        axis.add_patch(p)

        non_null = sampled != 0

        plt.scatter(ctr[0] + 0.5 * size, ctr[1] + 0.5 * size, c="r")
        plt.scatter(
            Xs.compress(IB) + 0.5 * size, Ys.compress(IB) + 0.5 * size, c="white", alpha=0.5, s=0.2
        )
        plt.scatter(
            Xs_snap.compress(IB * non_null) + 0.5,
            Ys_snap.compress(IB * non_null) + 0.5,
            c="r",
            alpha=0.5,
            s=0.4,
        )

        fig = plt.gcf()
        fig.savefig("debug_rt")

        # input('enter to continue')
        sleep(2)
        plt.close()

        #    plt.scatter(Xs[OOB],Ys[OOB],c="red",alpha=0.5,marker="x")
    plt.show()

    return np.sum(rays * weights)

def sum_over_rays_bias_multid(
    field, ctr, r200, rad_res, X_primes, Y_primes, Z_primes, debug=False
):

    """
    assume several fields
    compute sum over rays centred at ctr using given resoluttion
    field is a box to sum

    corrects for angular bias

    """

    nfields = np.shape(field)[0]
    size = np.shape(field)[1]

    ctr = np.asarray(ctr) - 0.5 * size
    Xs, Ys, Zs = [X_primes, Y_primes, Z_primes] + ctr[:, np.newaxis, np.newaxis]

    Rs = np.linalg.norm([Xs, Ys, Zs], axis=0)

    # used for getting data ... Need to be between 0 and size !!!
    Xs_snap, Ys_snap, Zs_snap = np.int32(
        [Xs + 0.5 * size, Ys + 0.5 * size, Zs + 0.5 * size]
    )
    # Rs_start = np.linalg.norm(
    #     [Xs_snap - ctr[0], Ys_snap - ctr[1], Zs_snap - ctr[2]], axis=0
    # )

    # print(Xs_snap,Ys_snap,Zs_snap)

    IB = Rs <= r200  # if points in r200 ...
    # OOB = ~IB

    # print(list(map(lambda x : (np.min(x),np.max(x)),[Xs_snap[IB],Ys_snap[IB],Zs_snap[IB]])))

    # print(IB, np.shape(field), np.shape(Xs))
    whs_IB = np.where(IB)

    sampled = np.zeros((nfields, Xs.shape[0], Xs.shape[1]))

    # print(nfields, np.prod(Xs.shape))

    # print(np.shape(field[:,Zs_snap[IB], Ys_snap[IB], Xs_snap[IB]]))
    # print(len(whs_IB), whs_IB)
    # print(np.shape(sampled))
    # print(np.shape(sampled[:, whs_IB]))

    sampled[:, whs_IB[0], whs_IB[1]] = field[:,Zs_snap[IB], Ys_snap[IB], Xs_snap[IB]]

    # if paths
    x_matches = np.zeros((np.shape(Xs)[0]), dtype=np.float32)
    y_matches = np.copy(x_matches)
    z_matches = np.copy(y_matches)

    argmin = np.argmin(np.abs(Rs[:, :] - r200), axis=1)

    inds = np.arange(len(x_matches))
    x_matches[inds] = Xs[inds, argmin]
    y_matches[inds] = Ys[inds, argmin]
    z_matches[inds] = Zs[inds, argmin]

    # if norms are 0 then we are on the border and the result should be 1
    # scal[np.isnan(scal)]=1

    scal = (
        (x_matches - ctr[0]) * x_matches
        + (y_matches - ctr[1]) * y_matches
        + (z_matches - ctr[2]) * z_matches
    ) / (
        (
            np.linalg.norm([x_matches, y_matches, z_matches], axis=0)
            * np.linalg.norm(
                [x_matches - ctr[0], y_matches - ctr[1], z_matches - ctr[2]], axis=0
            )
        )
    )

    scal[np.isnan(scal)] = 1
    scal[np.isinf(scal)] = 1
    # np.nan_to_num(scal,nan=1,posinf=1,neginf=1)

    # print(sampled.shape, rad_res, scal.shape)

    rays = np.exp(-np.sum(sampled, axis=2) * rad_res * scal)


    weights = (Rs[inds, argmin] ** -2) / np.sum(Rs[inds, argmin] ** -2)

    # print(np.sum(np.isinf(scal)),np.sum(np.isinf(rays)),np.sum(np.isinf(weights)),(np.sum(sampled,axis=1)*rad_res*scal)[np.isinf(rays)],(scal)[np.isinf(rays)],flush=True)

    if debug:  # r200>10:

        plt.figure(figsize=(10, 10))
        plt.subplot(111)
        plt.title("")
        plt.grid(True)
        # print(field[0,int(ctr[2]+0.5*size),:,:])
        # print(-np.sum(sampled,axis=1)*rad_res*scal)
        # plt.imshow(np.log10(field[0,int(ctr[2]+0.5*size),:,:]*rad_res),origin='lower',extent=[0,size,0,size])
        plt.imshow(
            np.log10(np.sum(field[0,:, :, :] * rad_res, axis=0)),
            origin="lower",
            extent=[0, size, 0, size],
        )

        plt.colorbar()

        p = pat.Circle(
            (0.5 * size, 0.5 * size),
            r200,
            color="magenta",
            fill=False,
            edgecolor="k",
            linewidth=2,
        )
        axis = plt.gca()
        axis.add_patch(p)

        non_null = sampled != 0

        plt.scatter(ctr[0] + 0.5 * size, ctr[1] + 0.5 * size, c="r")
        plt.scatter(
            Xs.compress(IB) + 0.5 * size, Ys.compress(IB) + 0.5 * size, c="white", alpha=0.5, s=0.2
        )
        plt.scatter(
            Xs_snap.compress(IB * non_null) + 0.5,
            Ys_snap.compress(IB * non_null) + 0.5,
            c="r",
            alpha=0.5,
            s=0.4,
        )

        fig = plt.gcf()
        fig.savefig("debug_rt")

        # input('enter to continue')
        sleep(2)
        plt.close()

        #    plt.scatter(Xs[OOB],Ys[OOB],c="red",alpha=0.5,marker="x")
    plt.show()

    # print(rays.shape, weights.shape)

    return np.sum(rays * weights[np.newaxis,:], axis=1)
    

def sum_over_rays_bias(
    field, ctr, r200, rad_res, X_primes, Y_primes, Z_primes, debug=False
):

    """
    compute sum over rays centred at ctr using given resoluttion
    field is a box to sum

    corrects for angular bias

    """

    size = np.shape(field)[0]

    ctr = np.asarray(ctr) - 0.5 * size
    Xs, Ys, Zs = [X_primes, Y_primes, Z_primes] + ctr[:, np.newaxis, np.newaxis]

    Rs = np.linalg.norm([Xs, Ys, Zs], axis=0)

    # used for getting data ... Need to be between 0 and size !!!
    Xs_snap, Ys_snap, Zs_snap = np.int32(
        [Xs + 0.5 * size, Ys + 0.5 * size, Zs + 0.5 * size]
    )
    # Rs_start = np.linalg.norm(
    #     [Xs_snap - ctr[0], Ys_snap - ctr[1], Zs_snap - ctr[2]], axis=0
    # )

    # print(Xs_snap,Ys_snap,Zs_snap)

    IB = Rs <= r200  # if points in r200 ...
    # OOB = ~IB

    # print(list(map(lambda x : (np.min(x),np.max(x)),[Xs_snap[IB],Ys_snap[IB],Zs_snap[IB]])))

    # print(IB, np.shape(field), np.shape(Xs))

    sampled = np.zeros_like(Xs)
    sampled[IB] = field[Zs_snap[IB], Ys_snap[IB], Xs_snap[IB]]

    # if paths
    x_matches = np.zeros((np.shape(Xs)[0]), dtype=np.float32)
    y_matches = np.copy(x_matches)
    z_matches = np.copy(y_matches)

    argmin = np.argmin(np.abs(Rs[:, :] - r200), axis=1)

    inds = np.arange(len(x_matches))
    x_matches[inds] = Xs[inds, argmin]
    y_matches[inds] = Ys[inds, argmin]
    z_matches[inds] = Zs[inds, argmin]

    # if norms are 0 then we are on the border and the result should be 1
    # scal[np.isnan(scal)]=1

    scal = (
        (x_matches - ctr[0]) * x_matches
        + (y_matches - ctr[1]) * y_matches
        + (z_matches - ctr[2]) * z_matches
    ) / (
        (
            np.linalg.norm([x_matches, y_matches, z_matches], axis=0)
            * np.linalg.norm(
                [x_matches - ctr[0], y_matches - ctr[1], z_matches - ctr[2]], axis=0
            )
        )
    )

    scal[np.isnan(scal)] = 1
    scal[np.isinf(scal)] = 1
    # np.nan_to_num(scal,nan=1,posinf=1,neginf=1)

    rays = np.exp(-np.sum(sampled, axis=1) * rad_res * scal)

    weights = (Rs[inds, argmin] ** -2) / np.sum(Rs[inds, argmin] ** -2)

    # print(np.sum(np.isinf(scal)),np.sum(np.isinf(rays)),np.sum(np.isinf(weights)),(np.sum(sampled,axis=1)*rad_res*scal)[np.isinf(rays)],(scal)[np.isinf(rays)],flush=True)

    if debug:  # r200>10:

        plt.figure(figsize=(10, 10))
        plt.subplot(111)
        plt.title("")
        plt.grid(True)
        # print(field[int(ctr[2]+0.5*size),:,:])
        # print(-np.sum(sampled,axis=1)*rad_res*scal)
        # plt.imshow(np.log10(field[int(ctr[2]+0.5*size),:,:]*rad_res),origin='lower',extent=[0,size,0,size])
        plt.imshow(
            np.log10(np.sum(field[:, :, :] * rad_res, axis=0)),
            origin="lower",
            extent=[0, size, 0, size],
        )

        plt.colorbar()

        p = pat.Circle(
            (0.5 * size, 0.5 * size),
            r200,
            color="magenta",
            fill=False,
            edgecolor="k",
            linewidth=2,
        )
        axis = plt.gca()
        axis.add_patch(p)

        non_null = sampled != 0

        plt.scatter(ctr[0] + 0.5 * size, ctr[1] + 0.5 * size, c="r")
        plt.scatter(
            Xs.compress(IB) + 0.5 * size, Ys.compress(IB) + 0.5 * size, c="white", alpha=0.5, s=0.2
        )
        plt.scatter(
            Xs_snap.compress(IB * non_null) + 0.5,
            Ys_snap.compress(IB * non_null) + 0.5,
            c="r",
            alpha=0.5,
            s=0.4,
        )

        fig = plt.gcf()
        fig.savefig("debug_rt")

        # input('enter to continue')
        sleep(2)
        plt.close()

        #    plt.scatter(Xs[OOB],Ys[OOB],c="red",alpha=0.5,marker="x")
    plt.show()

    return np.sum(rays * weights)
    

#
@njit(nogil=True, parallel=True, cache=True)
def sum_over_rays_bias_nopython(
    field, stt, r200, rad_res, rs, phis, thes, debug=False
):

    """
    compute sum over rays centred at stt using given resoluttion
    field is a box to sum

    corrects for angular bias

    """

    centre_start = (stt[0]==r200) and (stt[1]==r200) and (stt[2]==r200)

    ndim = 3
    size = field.shape

    box_stt = [0, 0, 0]

    result = 0.0

    lrs = len(rs)
    lps = len(phis)
    lts = len(thes)

    ray_ints = np.zeros((lps,lts))

    weight_sum = 0

    for idim in range(ndim):
        box_stt[idim] = stt[idim] - 0.5 * size[idim]

    for ip in range(lps):
        for it in range(lts):
            
            # x0,y0,z0 = sph_2_cart(r200, phis[ip], thes[it])

            # xedge = x0 + r200
            # yedge = y0 + r200
            # zedge = z0 + r200
            weight = 1
            reval = 1

            for ir in range(lrs):

                x0,y0,z0 = sph_2_cart_numba(rs[ir], phis[ip], thes[it])

                r0 = (x0**2+
                y0**2+
                z0**2)**0.5
                
                xsamp = x0 + stt[0]
                ysamp = y0 + stt[1]
                zsamp = z0 + stt[2]    

                xeval = x0 + stt[0] - r200
                yeval = y0 + stt[1] - r200
                zeval = z0 + stt[2] - r200

                weight = reval
                if ir==0 or centre_start:
                    scal=1
                else:
                    scal = ((x0 * xeval + 
                    y0 * yeval + 
                    z0 * zeval) /
                    (r0 * reval))

                reval = (xeval**2 + 
                yeval**2 + 
                zeval**2)**0.5

                # print(x0, y0, z0)
                # print(xsamp, ysamp, zsamp)
                # print(rs[ir], reval)

                # print(x, (r200 - ctr[0]))

                # xout = x<0 or x>2 * r200
                # yout = y<0 or y>2 * r200
                # zout = z<0 or z>2 * r200

                if reval>r200 : break
            
                ray_ints[ip, it] += field[int(zsamp//1), int(ysamp//1), int(xsamp//1)]
                # print(ray_ints[ip,it])


            weight_sum += weight**-2
            ray_ints[ip, it] *= weight * scal


    for ip in range(lps):
        for it in range(lts):
                result = result + ray_ints[ip, it] / weight_sum * rad_res
    # print(ray_ints)

    return result / (lps * lts)


def sum_over_rays_angle(field, ctr, r200, rad_res, X_circ, Y_circ, Z_circ):

    """
    compute sum over rays centred at ctr using given resoluttion
    field is a box to sum

    This verstion takes centre and sph points and computes paths (instead of
    taking paths by ref change), this ensures that all measurements are at
    the same posittions on the r200 sphere


    Euh .... this seems not to work or to give weird fesc results ... Does it though ???

    """

    # assert False, "this is broken ... makes fescs increase with mass ... ???"

    size = np.shape(field)[0]

    box_ctr = np.asarray([0.5 * size] * 3)

    ctr = np.asarray(ctr)

    delta_R = np.copy(ctr) - box_ctr

    X_primes, Y_primes, Z_primes = [X_circ, Y_circ, Z_circ] + delta_R[:, np.newaxis]
    # These are pos on sph viewed from studied cell centre

    Rs, Phis, Thes = cart_2_sph(X_primes, Y_primes, Z_primes)

    # get longest path
    longest = int(np.ceil(np.max(Rs) / rad_res)) + 1

    rays = np.zeros((len(X_primes), longest))

    # #print(np.shape(rays))
    # plt.figure(figsize=(10,10))
    # plt.subplot(111)
    # plt.title("")
    # plt.grid(True)
    # plt.imshow(np.log10(field[int(box_ctr[2]),:,:]*rad_res),origin='lower',extent=[0,size,0,size])
    # plt.colorbar()

    # p=pat.Circle((0.5*size,0.5*size),r200,color="magenta",fill=False,edgecolor='k',linewidth=2)
    # axis=plt.gca()
    # axis.add_patch(p)

    # plt.scatter(X_primes,Y_primes,c="cyan",alpha=0.5,s=0.2)
    # plt.scatter(X_circ,Y_circ,c="k",alpha=0.5,s=0.4)

    for i_path, (R, X_prime, Y_prime, Z_prime) in enumerate(
        zip(Rs, X_primes, Y_primes, Z_primes)
    ):

        # print(i_path)
        R_vect = np.arange(0, R + rad_res, rad_res)
        Phi = np.arctan2(Y_prime, X_prime)
        The = np.arccos(Z_prime / R) + np.pi
        lr = len(R_vect)
        X_paths, Y_paths, Z_paths = (
            np.asarray(sph_2_cart(R_vect, Phi, The)) + ctr[:, np.newaxis]
        )

        # plt.scatter(X_paths,Y_paths,c="brown",alpha=0.5,s=0.2)

        rays[i_path, :lr] = field[map(np.int32, [X_paths, Y_paths, Z_paths])]
        # print(rays[i_path,:lr])

    # plt.scatter(ctr[0],ctr[1],c='r')

    scal = (
        ((X_primes * X_circ) + (Y_primes * Y_circ) + (Z_primes * Z_circ))
        / np.linalg.norm([X_circ, Y_circ, Z_circ], axis=0)
        / Rs
    )

    scal[np.isnan(scal)] = 1

    rays_int = np.exp(-np.sum(rays, axis=1) * rad_res * scal)

    # print(rays_int,scal)
    # print(np.min(rays_int),np.min(rays_int*scal))

    # # #    plt.scatter(Xs[OOB],Ys[OOB],c="red",alpha=0.5,marker="x")
    # plt.show()
    # print(rays_int,scal)

    return rays_int


def cumsum_over_rays_angle(field, ctr, r200, rad_res, X_circ, Y_circ, Z_circ):

    """
    compute cum over rays centred at ctr using given resoluttion
    field is a box to sum

    This verstion takes centre and sph points and computes paths (instead of
    taking paths by ref change), this ensures that all measurements are at
    the same posittions on the r200 sphere

    Euh .... this seems not to work or to give weird fesc results ... Does is though ...

    """

    # assert False, "this is broken ... makes fescs increase with mass ... ???"

    size = np.shape(field)[0]

    box_ctr = np.asarray([0.5 * size] * 3)

    ctr = np.asarray(ctr)

    delta_R = np.copy(ctr) - box_ctr

    X_primes, Y_primes, Z_primes = [X_circ, Y_circ, Z_circ] + delta_R[:, np.newaxis]
    # These are pos on sph viewed from studied cell centre

    Rs, Phis, Thes = cart_2_sph(X_primes, Y_primes, Z_primes)

    # get longest path
    longest = int(np.ceil(np.max(Rs) / rad_res)) + 1

    rays = np.zeros((len(X_primes), longest))

    # #print(np.shape(rays))
    # plt.figure(figsize=(10,10))
    # plt.subplot(111)
    # plt.title("")
    # plt.grid(True)
    # plt.imshow(np.log10(field[int(box_ctr[2]),:,:]*rad_res),origin='lower',extent=[0,size,0,size])
    # plt.colorbar()

    # p=pat.Circle((0.5*size,0.5*size),r200,color="magenta",fill=False,edgecolor='k',linewidth=2)
    # axis=plt.gca()
    # axis.add_patch(p)

    # plt.scatter(X_primes,Y_primes,c="cyan",alpha=0.5,s=0.2)
    # plt.scatter(X_circ,Y_circ,c="k",alpha=0.5,s=0.4)

    for i_path, (R, X_prime, Y_prime, Z_prime) in enumerate(
        zip(Rs, X_primes, Y_primes, Z_primes)
    ):

        # print(i_path)
        R_vect = np.arange(0, R + rad_res, rad_res)
        Phi = np.arctan2(Y_prime, X_prime)
        The = np.arccos(Z_prime / R) + np.pi
        lr = len(R_vect)
        X_paths, Y_paths, Z_paths = (
            np.asarray(sph_2_cart(R_vect, Phi, The)) + ctr[:, np.newaxis]
        )

        # plt.scatter(X_paths,Y_paths,c="brown",alpha=0.5,s=0.2)

        rays[i_path, :lr] = field[map(np.int32, [X_paths, Y_paths, Z_paths])]
        # print(rays[i_path,:lr])

    # plt.scatter(ctr[0],ctr[1],c='r')

    scal = (
        ((X_primes * X_circ) + (Y_primes * Y_circ) + (Z_primes * Z_circ))
        / np.linalg.norm([X_circ, Y_circ, Z_circ], axis=0)
        / Rs
    )

    scal[np.isnan(scal)] = 1

    rays_int = np.exp(
        np.cumsum(-rays * rad_res, axis=1) * scal[:, np.newaxis]
    )  # cumulative sum of optical depths along the ray's axis

    # print(rays_int,scal)

    # # #    plt.scatter(Xs[OOB],Ys[OOB],c="red",alpha=0.5,marker="x")
    # plt.show()
    # print(rays_int,scal)

    return rays_int


def sum_over_rays_nexp(field, ctr, r200, rad_res, X_primes, Y_primes, Z_primes):

    """
    compute sum over rays centred at ctr using given resoluttion
    field is a box to sum

    """

    size = np.shape(field)[0]

    ctr = np.asarray(ctr) - 0.5 * size
    delta_R = np.copy(ctr)
    Xs, Ys, Zs = [X_primes, Y_primes, Z_primes] + delta_R[:, np.newaxis, np.newaxis]

    Rs = np.linalg.norm([Xs, Ys, Zs], axis=0)

    # used for getting data ... Need to be between 0 and size !!!
    Xs_snap, Ys_snap, Zs_snap = np.int32(
        [Xs + 0.5 * size, Ys + 0.5 * size, Zs + 0.5 * size]
    )

    IB = Rs <= r200  # if points in r200 ...
    OOB = ~IB

    # print(np.shape(OOB),np.shape(sampled))
    sampled = np.zeros_like(Xs)
    sampled[IB] = field[Xs_snap[IB], Ys_snap[IB], Zs_snap[IB]]

    Rs = np.linalg.norm([Xs, Ys, Zs], axis=0)

    # if paths
    x_matches = np.zeros((np.shape(Xs)[0]), dtype=np.float32)
    y_matches = np.copy(x_matches)
    z_matches = np.copy(y_matches)

    argmin = np.argmin(np.abs(Rs[:, :] - r200), axis=1)

    inds = np.arange(len(x_matches))
    x_matches[inds] = Xs[inds, argmin]
    y_matches[inds] = Ys[inds, argmin]
    z_matches[inds] = Zs[inds, argmin]

    # if norms are 0 then we are on the border and the result should be 1
    # scal[np.isnan(scal)]=1

    scal = (
        (x_matches - ctr[0]) * x_matches
        + (y_matches - ctr[1]) * y_matches
        + (z_matches - ctr[2]) * z_matches
    ) / (
        (
            np.linalg.norm([x_matches, y_matches, z_matches], axis=0)
            * np.linalg.norm(
                [x_matches - ctr[0], y_matches - ctr[1], z_matches - ctr[2]], axis=0
            )
        )
    )

    scal[np.isnan(scal)] = 1

    rays = np.sum(sampled, axis=1) * rad_res * scal

    return rays


def over_rays(field, ctr, r200, X_primes, Y_primes, Z_primes):

    """
    sample rays centred at ctr using given resoluttion
    field is a box to sample

    """

    size = np.shape(field)[0]

    ctr = np.asarray(ctr) - 0.5 * size
    delta_R = np.copy(ctr)
    Xs, Ys, Zs = [X_primes, Y_primes, Z_primes] + delta_R[:, np.newaxis, np.newaxis]

    Rs = np.linalg.norm([Xs, Ys, Zs], axis=0)

    # used for getting data ... Need to be between 0 and size !!!
    Xs_snap, Ys_snap, Zs_snap = np.int32(
        [Xs + 0.5 * size, Ys + 0.5 * size, Zs + 0.5 * size]
    )

    IB = Rs <= r200  # if points in r200 ...

    sampled = np.zeros_like(Xs)
    sampled[IB] = field[Xs_snap[IB], Ys_snap[IB], Zs_snap[IB]]

    return sampled
