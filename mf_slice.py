# -*- coding: utf-8 -*-
import sys,os

import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as cl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import libmf.micflash as mflash
import libmf.micplot as mplot
from var_settings_v3 import var_settings, intvar, GetVarSettings


#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

# ==== CONSTANTS ===============================================================
version_str = 'cfslice711b'
from constants import *

# ==== FILE SETTINGS ===========================================================
var_ch = mflash.var_ch5
basename = '*hdf5*_'
cntrange = list(range(34,100)) + list(range(1000,1100))

# ==== FIGURE SETTINGS =========================================================
fs = mplot.set_fig_preset('display')
mplot.fig_outdir = 'simmov/%s-%s'%(version_str, mplot.fig_target)

# ==== SLICE SETTINGS ==========================================================
point = np.array([-2.75,-14.75,11.75])*pc
#point = np.array([5.56236,2.43744,11.9372])*pc
radius = 4e+18

#stdkeys = ['temp','abund_h2','abund_co','cdto',]#'eint','lcdi','t_ff']
stdkeys = ['dens','temp','pres','abund_co']#'eint','lcdi','t_ff']

nguard = 0
grid_interpolation = 'none'#'bilinear'
imshow_interpolation = None#'bicubic'
res = (None, None, None)
x_extent = [point[0]-radius, point[0]+radius]
y_extent = [point[1]-radius, point[1]+radius]
z_extent = [point[2]-radius, point[2]+radius]


#===============================================================================
# ==== SLICE IMAGER ============================================================
#===============================================================================

ax_xy_order = {0:[1,2], 1:[0,2], 2:[0,1]}
ax_label = [r'x [pc]', r'y [pc]', r'z [pc]']

def axVarSlice(ax, dgrid, var, axis=2, offset=0.,
            x_extent=None, y_extent=None, z_extent=None, norm=None,
            show_cbar=True, show_axlabel=True, show_title=True):
            
    if dgrid is None:
        return None, None
    #
    units, varnorm, hist_cmap, varlabel, ulabel, vartitle, weighvar = GetVarSettings(var)
    barlabel = vartitle  +' ' +varlabel +' [' +ulabel +']'
    
    if norm is None:
        norm = varnorm
    #
    im_data = dgrid.read_slice(axis, offset=offset,
        interpolation=grid_interpolation,
        x_extent=x_extent, y_extent=y_extent, z_extent=z_extent,
        xres=res[0], yres=res[1], zres=res[2]) / units

    if show_title:
        slice_title = 'Slice, '+['x','y','z'][axis]+'= %.3g pc'%(offset/pc)
        ax.set_title(vartitle+' ('+slice_title+')')

    # Read simulation domain boundary coordinates
    extent = dgrid.fgrid.extent(axis) / pc
    ax_x, ax_y = ax_xy_order[axis]
    if (x_extent, y_extent, z_extent)[ax_x] is not None:
        extent[0] = np.array((x_extent, y_extent, z_extent))[ax_x] / pc
    if (x_extent, y_extent, z_extent)[ax_y] is not None:
        extent[1] = np.array((x_extent, y_extent, z_extent))[ax_y] / pc
    #
    im = ax.imshow(im_data.T, origin="lower", cmap=hist_cmap, norm=norm,
    interpolation=imshow_interpolation, extent=extent.flatten())
    ax.set(xlim=extent[0], ylim=extent[1])
    #
    if show_axlabel:
        ax.set(xlabel=ax_label[ax_x], ylabel=ax_label[ax_y])
    #
    if show_cbar==True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, label=barlabel)
    elif show_cbar==False:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("left", size="4%", pad=0.1)
        cbar = None
        cax.set_axis_off()
    elif show_cbar==None:
        cbar = None
    else:
        cax = show_cbar
        cbar = plt.colorbar(im, cax=cax, label=barlabel)
    return im, cbar


#===============================================================================
# ==== TEST ====================================================================  
#===============================================================================

def LoadVar(hdfdata, octree, key_in):
    var = key_in.split('::')[0]
    if var in ['lvdi', 'lvdi_sq', 'lcdi']:
        import localdisp_021 as localdisp
        lvdi = localdisp.get_lvdi(hdfdata, octree)
        hdfdata.cache('lvdi', lvdi)
        hdfdata.cache('lvdi_sq', lvdi**2)
        if var in ['lcdi',]:
            c_s = hdfdata['c_s']
            hdfdata.cache('lcdi', lvdi/c_s)
    elif var in ['xcoord', 'ycoord', 'zcoord', 'coords']:
        coords = octree.coords()
        hdfdata.cache('xcoord', coords[:,0,:,:,:])
        hdfdata.cache('ycoord', coords[:,1,:,:,:])
        hdfdata.cache('zcoord', coords[:,2,:,:,:])


#===============================================================================
# ==== MAIN ====================================================================  
#===============================================================================

def main(plotfile, y_var):

    dirname  = os.path.dirname(plotfile)
    filename = os.path.basename(plotfile)
    simname  = os.path.basename(dirname)
    subpath = simname
    
    outfilepath = mplot.get_savefig_filepath(filename, subpath)
    if os.path.exists(outfilepath):
        print(f'Skipping {filename}')
        return

    if y_var is None:
        varkeys = stdkeys
    else:
        varkeys = [y_var,]
        
    hdfdata = mflash.plotfile(plotfile)
    hdfdata.learn(mflash.var_mhd)
    hdfdata.learn(mflash.var_grid)
    hdfdata.learn(mflash.var_ch5)
    octree = mflash.pm3dgrid(hdfdata)
    
    nstep = hdfdata['integer scalars/nstep']
    simtime = hdfdata['real scalars/time'] * (1./Myr)
    simdt = hdfdata['real scalars/dt'] * (1./Myr)
    
    extparam = dict(x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
      
    n = len(varkeys)
    
    strdata = (filename,nstep,simtime,simdt)
    figtitle = '%s (step %i)\n(t= %2.6f Myr | dt= %.6f Myr)'%strdata

    if n==1:
        fig, ax = plt.subplots(1, 3)
        fig.suptitle(figtitle)
        key = varkeys[0]
        LoadVar(hdfdata, octree, key)
        dgrid = mflash.datagrid(hdfdata[key], octree, ng=nguard, verbose=True)
        for a in range(3):
            axVarSlice(ax[a], dgrid, key, axis=a, offset=point[a], **extparam)
            #rgr.axBlockSlice(ax[a], hdfdata, octree, axis=a, offset=point[a], **extparam)
    else:
        fig, ax = plt.subplots(3, n)
        fig.suptitle(figtitle)
        for i,key in enumerate(varkeys):
            dgrid = mflash.datagrid(hdfdata[key], octree, ng=nguard, verbose=True)
            LoadVar(hdfdata, octree, key)
            for a in range(3):
                axVarSlice(ax[a,i], dgrid, key, axis=a, offset=point[a], **extparam)
                #rgr.axBlockSlice(ax[a,i], hdfdata, octree, axis=a, offset=point[a], **extparam)
    
    outfilepath = mplot.savefig(fig, filename, subpath)
    
    hdfdata.close()


#===============================================================================
# ==== LOBBY ====================================================================
#===============================================================================

def getarg(i, default):
    try:
        return sys.argv[i]
    except:
        return default
        
if __name__ == '__main__':
    path = sys.argv[1]
    y_var = getarg(2, None)
    if os.path.isfile(path):
        main(path, y_var)
    elif os.path.isdir(path):
        cnt, cntfiles = mplot.countbatch(path, basename, cntrange, cntdigits=4)
        procf = lambda fn: main(fn, y_var)
        mplot.batchprocess(procf, cntfiles, verbose=True, strict=True)
    else:
        raise IOError('Path not understood.')

