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


def density_plot_fancy(stat,nbs,binsx,binsy,xlabel='',ylabel='',cax_label='Counts',xlog=True,ylog=True,collog=True,**kwargs):


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
