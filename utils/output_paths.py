import os


def gen_paths(sim_name,out_nb,suffix,assoc_mthd=''):


    out = os.path.join('/gpfs/alpine/proj-shared/ast031/jlewis/',sim_name+'_analysis')
    
    if assoc_mthd=='' or assoc_mthd=='fof_ctr':
    
        assoc_out=os.path.join(out,('assoc_halos_%s'%out_nb)+suffix)
        analy_out=os.path.join(out,('results_halos_%s'%out_nb)+suffix)

    else:

        assoc_out=os.path.join(out,('assoc_%s_halos_%s'%(assoc_mthd,out_nb))+suffix)
        analy_out=os.path.join(out,('results_%s_halos_%s'%(assoc_mthd,out_nb))+suffix)

        
    return(out,assoc_out,analy_out)