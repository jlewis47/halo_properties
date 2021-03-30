'''
------
Read phew/cats and star files and make associations
Saved as binary file that repeats for N phews and another that contains M stars
Associated stars and non associated stars are saved to separate binary files
Two files track each phew star number and 1st star file position
A final file saves phew IDs
------

This version includes halos that are close to each other and/or without stars


'''


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from read_phew import read_phew
from read_stars import read_stars
from scipy import spatial
import time
import string
import argparse
import os
from functions_latest import *
from output_paths import *
from read_fof import o_fof


def assoc_stars_to_haloes(out_nb,ldx,path,sim_name,use_fof=False):

        #phew_path='/data2/jlewis/dusts/output_00'+out_nb
        #star_path='/data2/jlewis/dusts/'
        #info_path='/data2/jlewis/dusts/output_00'+out_nb

        
        info_path=os.path.join(path,'output_00'+out_nb)
        star_path=path
        phew_path=os.path.join(path,'output_00'+out_nb)

        fof_suffix=''
        if use_fof:
                fof_path=os.path.join(phew_path,'fofres/halos_ll=0.2')
                fof_suffix='_fof'
                
        Np_tot=ldx**3

        '''Get scale factor and co'''
        (t,a,H0,om_m,om_l,om_k,om_b,unit_l,unit_d,unit_t,l,Lco,L,px_to_m) = get_infos_no_t(info_path,
                                                                                           out_nb,
                                                                                           ldx)

        ran = [-ldx,0,ldx]
        pos_vects = [[i,j,k] for k in ran for j in ran for i in ran]


        tstep=0.0 #temporary


        out = os.path.join(analy_path,sim_name)
        assoc_out=os.path.join(assoc_path,sim_name,('assoc_halos_%s'%out_nb)+fof_suffix)



        #if no out folder, make it
        if sim_name not in os.listdir(assoc_path):
            os.makedirs(assoc_out)

        #if no out folder, make it
        if ('assoc_halos_%s'%out_nb)+fof_suffix not in os.listdir(os.path.join(assoc_path,sim_name)):
            os.mkdir(assoc_out)



        #if no out folder, make it
        if sim_name not in os.listdir(analy_path):
            os.mkdir(out)




        Mp=3.*(H0*1e3/(pc*1e6))**2.*(om_m-om_b)*(Lco*pc*1e6)**3./Np_tot/8./np.pi/G/Msol #En Msol 4.04*1e5
        print("Partcle mass is :",Mp)
        if not  'Mp' in  os.listdir(out):
                out_Mp = open(os.path.join(out,'Mp'),'wb')
                out_Mp.write(np.float64(Mp))
                out_Mp.close()



            # print(np.all([x_cond,y_cond,z_cond],axis=0) == np.all(np.all([ctrs[:]<=test_pos-radius,ctrs[:]>=test_pos+radius],axis=0),axis=1))

            #return(np.all(np.all([ctrs[:]<=test_pos-radius,ctrs[:]>=test_pos+radius],axis=0),axis=1))

        #phews
        if not use_fof:
                print('Reading phews')
                phews=read_phew(phew_path)
                #convert to same format as phews id,mass,x,y,z; mass is in particle masses
                phews=phews[:,[0,6,2,3,4]]
                phews[:,1]=phews[:,1]*ldx**3*Mp
        else:
                print('Reading fofs')
                phews=o_fof(fof_path)
                phews[:,1]*=Mp



        #Stars
        stars=read_stars(star_path,out_nb)

        stars=np.transpose([stars['msol'],stars['x'],stars['y'],stars['z'],stars['Age'],stars['Z']])

        #print(np.shape(stars))

        star_tree = spatial.cKDTree(stars[:,1:4]*(ldx))

        #Local star IDs for creating list of 'lone stars'
        star_ids = np.arange(len(stars))



        size=np.asarray(np.int32(len(phews)))


        stt = time.time()

        tot_nb_stars=0



        #Association and output
        print('Association starting')
        out_halos = open(os.path.join(assoc_out,('assoc_halos_%s' %out_nb)+fof_suffix),'wb')
        out_halosST_tot = open(os.path.join(assoc_out,('assoc_halosST_tot_%s' %out_nb)+fof_suffix),'wb')
        out_halosST = open(os.path.join(assoc_out,('assoc_halosST_%s' %out_nb)+fof_suffix),'wb')
        out_halosID = open(os.path.join(assoc_out,('assoc_halosID_%s' %out_nb)+fof_suffix),'wb')
        out_stars = open(os.path.join(assoc_out,('assoc_stars_%s' %out_nb)+fof_suffix),'wb')
        out_halosST_IDs = open(os.path.join(assoc_out,('assoc_starsIDs_%s' %out_nb)+fof_suffix),'wb')
        out_lone_stars = open(os.path.join(assoc_out,('assoc_lone_stars_%s' %out_nb)+fof_suffix),'wb')

        out_halos.write(size)

        not_lone_star_ids = []

        # fig=plt.figure()
        # ax=fig.add_subplot(111)
        # ax.set_xlim(0,1)
        # ax.set_ylim(0,1)
        # plt.show(block=False)

        #sm_phews=phews[phews[:,0]==12219549164]

        #for phew_nb,phew in enumerate(sm_phews):
        for phew_nb,phew in enumerate(phews):



            r_px = get_r200(phew[1]/Mp)
            #print(r_px)


            ctr_vanilla = phew[2:5] #ctr without rounding
            ctr = do_half_round(phew[2:5]*(ldx))

            #Find nearby stars
            #big_ball = star_tree.query_ball_point(ctr,r_px)
            #this line checks all edge reflexions    
            big_ball=np.int32(get_27_tree(ctr,star_tree,r_px,pos_vects))
            # l_bb=np.shape(big_ball)

            phew_stars=stars[big_ball,:]
            nb_stars = (len(phew_stars))
            nb_yng_stars = (len(phew_stars[phew_stars[:,-2]<10.+tstep]))


            # if np.sum(l_bb)>0:
            #     if np.sum(l_bb)-np.sum(np.shape(star_tree.query_ball_point(ctr,r_px)))!=0:
            #         print(l_bb,np.shape(star_tree.query_ball_point(ctr,r_px)))
            #         ax.scatter(phew_stars[:,1],phew_stars[:,2],s=5,alpha=0.5,c='k')
            #     ax.scatter(phew_stars[:,1],phew_stars[:,2],s=1,alpha=0.5)
            #     fig.canvas.draw()
            #     fig.canvas.flush_events()
            #     plt.pause(0.0001)    



            if nb_yng_stars == 1: # if only one star we should recentre
                """
                If only one star then re-centre halo on it
                """
                ctr_star = phew_stars[phew_stars[:,4]<10.+tstep,1:4][0]
                phews[phew_nb,2:5]=ctr_star
                ctr_vanilla = ctr_star #ctr without rounding
                ctr=do_half_round(ctr_star*(ldx))
                #big_ball = star_tree.query_ball_point(ctr,r_px) #Use fast method for ball that holds box
                #this line checks all edge reflexions
                big_ball=np.int32(get_27_tree(ctr,star_tree,r_px,pos_vects))
                phew_stars=stars[big_ball,:] #r200/side
                nb_stars = (len(phew_stars))
                nb_yng_stars = (len(phew_stars[phew_stars[:,-2]<10.+tstep]))







            idx = np.int64(phew[0])
            fmass = np.float32(phew[1])
            fx = np.float32(ctr_vanilla[0])
            fy = np.float32(ctr_vanilla[1])
            fz = np.float32(ctr_vanilla[2])
            fr200 = np.float32(r_px)
            sidx = np.int64(star_ids)[big_ball]


            smass = np.float32(phew_stars[:,0])
            sx = np.float32(phew_stars[:,1])
            sy = np.float32(phew_stars[:,2])
            sz = np.float32(phew_stars[:,3])
            age  =  np.float32(phew_stars[:,4])
            metal =  np.float32(phew_stars[:,5])



            for phew_star_ind in range(len(sidx)):
                     out_halosST_IDs.write(sidx[phew_star_ind])
                     out_stars.write(smass[phew_star_ind])
                     out_stars.write(sx[phew_star_ind])
                     out_stars.write(sy[phew_star_ind])
                     out_stars.write(sz[phew_star_ind])
                     out_stars.write(age[phew_star_ind])
                     out_stars.write(metal[phew_star_ind])             


            out_halosID.write(idx)
            out_halos.write(fmass)
            out_halos.write(fx)
            out_halos.write(fy)
            out_halos.write(fz)
            out_halos.write(fr200)


            out_halosST_tot.write(np.int64(tot_nb_stars))
            out_halosST.write(np.int64(nb_stars))
            tot_nb_stars+=nb_stars





            #Keep track of associated star IDs    
            not_lone_star_ids.append(sidx.tolist())

        print('Association Finished')




        #unique_stars_ids,unique_star_pos = (np.unique(np.concatenate(not_lone_star_ids),return_index=True))
        #unique_stars = stars
        #unique_stars[unique_star_pos]=np.zeros(np.shape(unique_stars[unique_star_pos]))


            # else:
            #     out_stars.write('0.0')
            #     out_stars.write('0.0')
            #     out_stars.write('0.0')
            #     out_stars.write('0.0')
            #     out_stars.write('0.0')
            #     out_stars.write('0.0')        



        out_stars.close()
        out_halos.close()
        out_halosID.close()
        out_halosST.close()
        out_halosST_IDs.close()
        out_halosST_tot.close()

        #Write unassigned stars
        star_ids = set(star_ids)
        not_lone_star_ids = set(np.concatenate(not_lone_star_ids))

        lone_star_ids = np.asarray(list(star_ids-not_lone_star_ids))
        lone_stars = stars[np.int64(lone_star_ids),:]
        out_lone_stars.write(bytearray(np.float32(lone_stars[:,:])))
        out_lone_stars.close()

        print(len(not_lone_star_ids),len(lone_star_ids))

        out= open(os.path.join(assoc_path,sim_name,('assoc_dust_%s_star_nb' %out_nb)+fof_suffix),'wb')
        out.write(np.int64(tot_nb_stars))
        out.close()



        


"""
Main body
"""


if __name__ =='__main__':

        Arg_parser = argparse.ArgumentParser('Associate stars and halos in full simulation')

        Arg_parser.add_argument('nb',metavar='nsnap',type=str,help='snap number string, give as "XXX" so that 00001 is "001" and 00111 is "111"')
        Arg_parser.add_argument('ldx',metavar='ldx',type=str,help='box size in cells')
        Arg_parser.add_argument('path',metavar='path',type=str,help='path to sim')
        Arg_parser.add_argument('simname',metavar='simname',type=str,help='sim name (will be used to create dirs)')

        args = Arg_parser.parse_args()


        out_nb = args.nb
        ldx=int(args.ldx)
        path=args.path
        sim_name=args.simname

    
        assoc_stars_to_haloes(out_nb,ldx,path,sim_name)
    
