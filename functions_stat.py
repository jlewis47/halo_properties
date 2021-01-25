"""
Essentially wrappers for scipy stats stuff
"""

import numpy as np
from scipy.stats import binned_statistic,binned_statistic_2d
import matplotlib
from matplotlib import pyplot as plt
from functions_latest import make_figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

def get_25(array):
    values=np.sort(array)
    l=len(values)
    return(values[int(0.25*l)])

def get_75(array):
    values=np.sort(array)
    l=len(values)
    return(values[int(0.75*l)])


def mef_iqr(array):
    
    values=np.sort(array)
    l=len(values)

    inds=[int(fact*l) for fact in [0.25,0.5,0.75]]
    
    return(values[inds])    
    
def mef_iqr_binned_statistic(array,bins):
    
    nbs,bins,loc_bins=binned_statistic(array,array,'count',bins=bins)
    
    enough_data=nbs>5
    
    med,bins,loc_bins=binned_statistic(array,array,np.nanmedian,bins=bins)
    y25,bins,loc_bins=binned_statistic(array,array,get_25,bins=bins)
    y75,bins,loc_bins=binned_statistic(array,array,get_75,bins=bins)    
    
    
    
    return(med[enough_data],y25[enough_data],y75[enough_data])

def get_4stats(xbins,xdata,ydata):


     med,bins,loc_bins=binned_statistic(xdata,ydata,'median',bins=xbins)
     mean,bins,loc_bins=binned_statistic(xdata,ydata,'mean',bins=xbins)
     distr_25_occ,bins,loc_bins=binned_statistic(xdata,ydata,get_25,bins=xbins)
     distr_75_occ,bins,loc_bins=binned_statistic(xdata,ydata,get_75,bins=xbins)


     return(mean,med,distr_25_occ,distr_75_occ)


def get_4_stats_2d(xbins,zbins,xdata,ydata,zdata):

     mean_ydata_2d,binsx,binsy,loc_bins=binned_statistic_2d(xdata,zdata,ydata,'mean',bins=[xbins,zbins])
     median_ydata_2d,binsx,binsy,loc_bins=binned_statistic_2d(xdata,zdata,ydata,'median',bins=[xbins,zbins])
     ydata_2d_25,binsx,binsy,loc_bins=binned_statistic_2d(xdata,zdata,ydata,get_25,bins=[xbins,zbins])
     ydata_2d_75,binsx,binsy,loc_bins=binned_statistic_2d(xdata,zdata,ydata,get_75,bins=[xbins,zbins])

     return(mean_ydata_2d,median_ydata_2d,ydata_2d_25,ydata_2d_75)


def density_plot_fancy(stat,nbs,binsx,binsy,xlabel='',ylabel='',cax_label='Counts',xlog=True,ylog=True,collog=True,
                       xhist_log=True,yhist_log=True,**kwargs):


    vmin=np.nanmin(stat)
    vmax=np.nanmax(stat)
    cmap='jet'



    for key,value in kwargs.items():
        if key=='vmin':
            vmin=value
        if key=='vmax':
            vmax=value
        if key=='cmap':
            cmap=value

            
    if collog:
        if vmin<=0:
            vmin=np.nanmin(stat[stat!=0])

    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='white')

    print('Found (vmin,vmax)=(%.1e,%.1e)'%(vmin,vmax))

    fig,ax=make_figure()

    hist_1d_ticks=[0.0,0.5,1.0]

    divider = make_axes_locatable(ax)
    hist_x_ax = divider.append_axes("top", size="22%", pad=0.125)
    hist_x_ax.xaxis.set_tick_params(bottom=False,top=True,labelbottom=False,labeltop=True)

    x_axis_plot=(np.sum(nbs,axis=1)/np.sum(nbs))
    hist_x_ax.plot((binsx[:-1]),x_axis_plot,drawstyle='steps-post')
    x_axis_ticks=np.round(100*np.asarray(hist_1d_ticks)*1.1*np.max(x_axis_plot))/100


    hist_x_ax.set_xlim((binsx[0]),(binsx[-1]))
    hist_x_ax.set_xlabel(xlabel)
    hist_x_ax.set_ylabel('PDF')
    hist_x_ax.xaxis.set_label_position('top')
    hist_x_ax.set_yticks(x_axis_ticks)
    if xlog : hist_x_ax.set_xscale('log')
    hist_x_ax.xaxis.set_tick_params(bottom=False,labelbottom=False,which='both')
    hist_x_ax.grid()

    hist_y_ax = divider.append_axes("left", size="22%", pad=0.125)
    hist_y_ax.xaxis.set_tick_params(left=True,labelleft=True,which='both')

    y_axis_plot=(np.sum(nbs,axis=0)/np.sum(nbs))
    hist_y_ax.plot(y_axis_plot,(binsy[:-1]),drawstyle='steps-pre')
    y_axis_ticks=np.round(100*np.asarray(hist_1d_ticks)*1.1*np.max(y_axis_plot))/100

    hist_y_ax.set_ylim((binsy[0]),(binsy[-1]))
    hist_y_ax.invert_xaxis()
    hist_y_ax.set_ylabel(ylabel)
    hist_y_ax.set_xlabel('PDF')
    hist_y_ax.set_xticks(y_axis_ticks)
    hist_y_ax.set_xticklabels(map(str,y_axis_ticks),rotation=45)
    if ylog : hist_y_ax.set_yscale('log')
    hist_y_ax.grid()

    hist_x_ax.xaxis.set_tick_params(bottom=False,top=False,labelbottom=False,labeltop=False,which='minor')

    ax.yaxis.set_tick_params(left=False,labelleft=False,which='both')
    ax.xaxis.set_tick_params(bottom=False,labelbottom=False,which='both')

    cax=divider.append_axes("right", size="5%", pad=0.125)

    if yhist_log : 
        hist_y_ax.set_xlim(1e-2,1)
        hist_y_ax.set_xscale('log')
    if xhist_log : 
        hist_x_ax.set_ylim(1e-2,1)
        hist_x_ax.set_yscale('log')        

    if collog:
            if xlog and ylog :
                    img=ax.imshow((stat.T),extent=np.log10([binsx[0],binsx[-1],binsy[0],binsy[-1]]),aspect='auto',origin='lower',norm=LogNorm(),vmin=vmin,vmax=vmax,cmap=cmap)
            elif xlog:
                    img=ax.imshow((stat.T),extent=[np.log10(binsx[0]),np.log10(binsx[-1]),binsy[0],binsy[-1]],aspect='auto',origin='lower',norm=LogNorm(),vmin=vmin,vmax=vmax,cmap=cmap)
            elif ylog:
                    img=ax.imshow((stat.T),extent=[binsx[0],binsx[-1],np.log10(binsy[0]),np.log10(binsy[-1])],aspect='auto',origin='lower',norm=LogNorm(),vmin=vmin,vmax=vmax,cmap=cmap)
            else:
                    img=ax.imshow((stat.T),extent=[binsx[0],binsx[-1],binsy[0],binsy[-1]],aspect='auto',origin='lower',norm=LogNorm(),vmin=vmin,vmax=vmax,cmap=cmap)
    else:
        if xlog and ylog :
            img=ax.imshow((stat.T),extent=np.log10([binsx[0],binsx[-1],binsy[0],binsy[-1]]),aspect='auto',origin='lower',vmin=vmin,vmax=vmax,cmap=cmap)
        elif xlog:
            img=ax.imshow((stat.T),extent=[np.log10(binsx[0]),np.log10(binsx[-1]),binsy[0],binsy[-1]],aspect='auto',origin='lower',vmin=vmin,vmax=vmax,cmap=cmap)
        elif ylog:
            img=ax.imshow((stat.T),extent=[binsx[0],binsx[-1],np.log10(binsy[0]),np.log10(binsy[-1])],aspect='auto',origin='lower',vmin=vmin,vmax=vmax,cmap=cmap)
        else:
            img=ax.imshow((stat.T),extent=[binsx[0],binsx[-1],binsy[0],binsy[-1]],aspect='auto',origin='lower',vmin=vmin,vmax=vmax,cmap=cmap)





    plt.colorbar(img,cax=cax)

    ax.grid()
    
    #cax.set_yscale('log')
    cax.set_ylabel(cax_label)
    cax.yaxis.set_label_position("right")
    cax.yaxis.set_tick_params(left=False,labelleft=False,right=True,labelright=True,which='both')
    #cax.yaxis.set_tick_params(left=False,labelleft=False,right=False,labelright=False,which='minor')

    #cax_ticks=cax.get_yticklabels()
    #cax.set_yticklabels(["%.1e"%10**float(tick.get_text()) for tick in cax_ticks])


    return(fig,ax,hist_x_ax,hist_y_ax,cax)
    #plt.show()    

    
def density_plot_corner(stat,binsx,binsy,fig,ax,xlabel='',ylabel='',cax_label='Counts',xlog=True,ylog=True,collog=True,**kwargs):


    vmin=np.nanmin(stat)
    vmax=np.nanmax(stat)
    cmap='jet'



    for key,value in kwargs.items():
        if key=='vmin':
            vmin=value
        if key=='vmax':
            vmax=value
        if key=='cmap':
            cmap=value

            
    if collog:
        if vmin<=0:
            vmin=np.nanmin(stat[stat!=0])

    current_cmap = matplotlib.cm.get_cmap()
    current_cmap.set_bad(color='white')

    #print('Found (vmin,vmax)=(%.1e,%.1e)'%(vmin,vmax))
    


    #divider = make_axes_locatable(ax)
    
    #cax=divider.append_axes("right", size="5%", pad=0.125)

    if ylog : 
        #print((binsy[0]),(binsy[-1]))
        ax.set_ylim((binsy[0]),(binsy[-1]))
        ax.set_yscale('log')
    #if xlog : 
    #    print((binsx[0]),(binsx[-1]))
    #    ax.set_xlim((binsx[0]),(binsx[-1]))
    #    ax.set_xscale('log')

    X,Y=np.meshgrid(binsx[:-1],binsy[:-1])   
    
    #print(X,Y)
    
    #print(np.shape(X),np.shape(stat))
        
    if collog:
        
            img=ax.pcolormesh(X,Y,stat.T,norm=LogNorm(),vmin=vmin,vmax=vmax,cmap=cmap)
            #img=ax.imshow((stat.T),extent=[binsx[0],binsx[-1],binsy[0],binsy[-1]],aspect='auto',origin='lower',norm=LogNorm(),vmin=vmin,vmax=vmax,cmap=cmap)

    else:
            img=ax.pcolormesh(X,Y,stat.T,vmin=vmin,vmax=vmax,cmap=cmap)
            #img=ax.imshow((stat.T),extent=[binsx[0],binsx[-1],binsy[0],binsy[-1]],aspect='auto',origin='lower',vmin=vmin,vmax=vmax,cmap=cmap)



    #plt.colorbar(img,cax=cax)

    ax.grid()
    
    #cax.set_yscale('log')
    #cax.set_ylabel(cax_label)
    #cax.yaxis.set_label_position("right")
    #cax.yaxis.set_tick_params(left=False,labelleft=False,right=True,labelright=True,which='both')
    #cax.yaxis.set_tick_params(left=False,labelleft=False,right=False,labelright=False,which='minor')
    
    #cax_ticks=cax.get_yticklabels()
    #cax.set_yticklabels(["%.1e"%10**float(tick.get_text()) for tick in cax_ticks])


    return(fig,ax)
    #plt.show()    
    
    
def corner_plot(datas,labels='none',bins='none'):
    
       
    Ndata=len(datas)


    
    fig,ax=plt.subplots(Ndata,Ndata,figsize=(25,25),
                        sharex='col',gridspec_kw={'hspace':.3,'wspace':.3})
           
    if labels!='none':
        
            assert len(labels)==len(datas), "Number of data vectors didn't match the number of label strings"        
    
    if bins=='none':
        #determine data bins
        bin_dx=0.25
        #Zscore_thresh=0.5
        bins=[]
        for ibin in range(Ndata):
        
            #Zscores=np.abs(scipy.stats.zscore(datas[ibin]))
        
            logmax=np.log10(datas[ibin].max())
            #logmin=np.log10(datas[ibin][Zscores<Zscore_thresh].min())
            logmin=np.log10(datas[ibin].min())
            
            #print(logmin,logmax)
            
            if np.isinf(logmin):
                #logmin=np.log10(datas[ibin][Zscores<Zscore_thresh][datas[ibin][Zscores<Zscore_thresh]>0].min())
                logmin=np.log10(datas[ibin][datas[ibin]>0].min())
            if np.isinf(logmax):
                logmax=np.log10(datas[ibin][datas[ibin]>0].max())
                
            logmin=np.floor(logmin/bin_dx)*bin_dx-bin_dx
            logmax=np.ceil(logmax/bin_dx)*bin_dx+bin_dx
                
            #print(logmin,logmax,bin_dx)
            
            cur_bin=10**np.arange(logmin,logmax+bin_dx,bin_dx)
    
            bins.append(cur_bin)
        
        else:
            
            assert len(bins)==len(datas), "Number of data vectors didn't match the number of bin vectors"
    
    for i in range(Ndata): #cols
        for j in range(Ndata): #rows
            
            cur_ax=ax[i,j]   
            
            #for axis in ['top','bottom','left','right']:
            #    cur_ax.spines[axis].set_linewidth(1.5)

            #print(i,j)
            
            if i<j:
                cur_ax.remove()
                continue
            
            
            cur_ax.tick_params(right='on',top='on',direction="in",which='major',axis='both')
            
            cur_ax.tick_params(labelright='off',labelbottom='off',labelleft='off',labeltop='off',
                               right='off',left='off',top='off',bottom='off',
                               direction="in",which='minor',axis='both')            
            
            
            
            #if i>0 :                
            #    cur_ax.tick_params(left='off',which='both',labelleft='off')
            #if j>0 :                
            #    cur_ax.tick_params(bottom='off',which='both',labelbottom='off')                
            
            
            if i!=j:#if not diag we do a density map
                #print()
                
                #print('map')
                nbs,binsx,binsy,whs=binned_statistic_2d(datas[j],datas[i],datas[i],'count',bins=[bins[j],bins[i]])                
                
                density_plot_corner(nbs,binsx,binsy,fig,cur_ax,
                                    xlog=True,ylog=True,collog=True)
                
                if j==0 and labels!='none':
                    
                    cur_ax.set_ylabel(labels[i])#,size=14)
                
            else:
                
                #print('hist')
                
                nbs,binsx=np.histogram(datas[i],bins=bins[i])
                
                #print(nbs)
                #print(binsx)
                
                cur_ax.plot(binsx[1:],nbs,drawstyle='steps-pre')
                cur_ax.set_yscale('log')
                cur_ax.set_xscale('log')
                
                cur_ax.set_ylabel('Counts')#,size=14)
                
                cur_ax.grid()
                             
            if i==Ndata-1 and labels!='none':
                    
                cur_ax.set_xlabel(labels[j])#,size=14)                    
                
    
    return(fig,ax)
        
