# -*- coding: utf-8 -*-
import sys,os

import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as cl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from operator import itemgetter

import libmf.micflash as mflash
import libmf.micplot as mplot
import libmf.tools as tools
from var_settings_v3 import var_settings, intvar, GetVarSettings


#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

# ==== CONSTANTS ===============================================================
version_str = 'cfslice801a'
from constants import *

# ==== FILE SETTINGS ===========================================================
var_ch = mflash.var_ch5
basename = '*hdf5*_'
cntrange = list(range(500))

# ==== FIGURE SETTINGS =========================================================
fs = mplot.set_fig_preset('display')
mplot.fig_outdir = 'slice/%s-%s'%(version_str, mplot.fig_target)

# ==== SLICE SETTINGS ==========================================================
point = np.array([0.,0.,0.])*au
radius = 4e+18

varsets = {
    'mhd': ['dens','temp','pres'],
    'vel': ['velx','vely','velz','vel'],
    'mag': ['magx','magy','magz','mag'],
    }
varset_default = varsets['mhd']

stdax = [2,0,1]
nguard = 0
grid_interpolation = 'none'#'bilinear'
imshow_interpolation = None#'bicubic'
res = (None, None, None)
x_extent = None # [point[0]-radius, point[0]+radius]
y_extent = None # [point[1]-radius, point[1]+radius]
z_extent = None # [point[2]-radius, point[2]+radius]


#===============================================================================
# ==== PLOTFILE HELPER =========================================================
#===============================================================================

def OpenPlotfile(plotfile):
   hdfdata = mflash.plotfile(plotfile)
   hdfdata.learn(mflash.var_mhd)
   hdfdata.learn(mflash.var_grid)
   hdfdata.learn(mflash.var_ch5)
   octree = mflash.pm3dgrid(hdfdata)
   return hdfdata, octree


#===============================================================================
# ==== SLICE HELPER ============================================================
#===============================================================================

ax_units = {'pc':pc, 'Au':au, 'km':1e+5, 'cm':1.}
usorter = dict(key=itemgetter(1), reverse=True)

def PickAxisUnit(extent, rth=.1):
   exmax = np.max(np.abs(extent))
   for axulabel, axunit in sorted(ax_units.items(), **usorter):
       if exmax > rth*axunit: break
   return axunit, axulabel
   

#===============================================================================
# ==== SLICE IMAGER ============================================================
#===============================================================================

ax_xy_order = {0:[1,2], 1:[0,2], 2:[0,1]}

def axVarSlice(ax, dgrid, var, axis=2, offset=0.,
            x_extent=None, y_extent=None, z_extent=None, norm=None,
            show_cbar=True, show_axlabel=True, show_title=True):
            
    if dgrid is None:
        return None, None
    #
    units, varnorm, hist_cmap, varlabel, ulabel, vartitle, weighvar = GetVarSettings(var)
    barlabel = f'{varlabel} [{ulabel}]'
    
    if norm is None:
        norm = varnorm
    #
    im_data = dgrid.read_slice(axis, offset=offset,
        interpolation=grid_interpolation,
        x_extent=x_extent, y_extent=y_extent, z_extent=z_extent,
        xres=res[0], yres=res[1], zres=res[2]) / units

    # Read simulation domain boundary coordinates
    extent_cgs = dgrid.fgrid.extent(axis)
    ax_x, ax_y = ax_xy_order[axis]
    if (x_extent, y_extent, z_extent)[ax_x] is not None:
        extent_cgs[0] = np.array((x_extent, y_extent, z_extent))[ax_x]
    if (x_extent, y_extent, z_extent)[ax_y] is not None:
        extent_cgs[1] = np.array((x_extent, y_extent, z_extent))[ax_y]
    #
    axunit, axulabel = PickAxisUnit(extent_cgs, rth=.1)
    extent = extent_cgs / axunit
    #
    im = ax.imshow(im_data.T, origin="lower", cmap=hist_cmap, norm=norm,
    interpolation=imshow_interpolation, extent=extent.flatten())
    ax.set(xlim=extent[0], ylim=extent[1])
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
    #
    if show_axlabel:
        axustr = ' [%s]'%axulabel
        ax.set(xlabel=['x','y','z'][ax_x]+axustr, ylabel=['x','y','z'][ax_y]+axustr)
    #
    if show_title:
        slice_title =  ['x','y','z'][axis]
        slice_title += '= %.3g %s'%(offset/axunit, axulabel)
        ax.set_title(vartitle+' ('+slice_title+')')
    #
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

def main(plotfile, varset_key, varspec=None, storefig=False):

    varkeys = varsets.get(varset_key, varset_default)
    if varspec is None:
        datakeys = varkeys
    else:
        datakeys = ['::'.join([k,varspec]) for k in varkeys]
        
    filename = os.path.basename(plotfile)
        
    if storefig: # Check if file was already made (and should be skipped)
        dirname  = os.path.dirname(plotfile)
        simname  = os.path.basename(dirname)
        subpath = f'{simname}/{varset_key}'
        outfilepath = mplot.get_savefig_filepath(filename, subpath)
        if os.path.exists(outfilepath):
            print(f'Skipping {filename}')
            return None
        
    hdfdata, octree = OpenPlotfile(plotfile)
    
    extparam = dict(x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
    
    n = len(datakeys)
    if n==1:
        fig, ax = plt.subplots(1, 3)
        key = datakeys[0]
        LoadVar(hdfdata, octree, key)
        dgrid = mflash.datagrid(hdfdata[key], octree, ng=nguard, verbose=True)
        aspect_accu = 0.
        for a in stdax:
            axVarSlice(ax[a], dgrid, key, axis=a, offset=point[a], **extparam)
            #rgr.axBlockSlice(ax[a], hdfdata, octree, axis=a, offset=point[a], **extparam)
            xlim = ax[a].get_xlim()
            ylim = ax[a].get_ylim()
            aspect = (xlim[1]-xlim[0]) / (ylim[1]-ylim[0])
            aspect_accu += aspect
    else:
        fig, ax = plt.subplots(3, n)
        for i,key in enumerate(datakeys):
            LoadVar(hdfdata, octree, key)
            dgrid = mflash.datagrid(hdfdata[key], octree, ng=nguard, verbose=True)
            aspect_accu = 0.
            for a in stdax:
                axVarSlice(ax[a,i], dgrid, key, axis=a, offset=point[a], **extparam)
                #rgr.axBlockSlice(ax[a,i], hdfdata, octree, axis=a, offset=point[a], **extparam)
                xlim = ax[a,i].get_xlim()
                ylim = ax[a,i].get_ylim()
                aspect = (xlim[1]-xlim[0]) / (ylim[1]-ylim[0])
                aspect_accu += aspect

    nstep = hdfdata['integer scalars/nstep']
    simtime = hdfdata['real scalars/time'] * (1e+3/Myr)
    simdt = hdfdata['real scalars/dt'] * (1e+3/Myr)
    figtitle =  '%s (step %i)'%(filename,nstep)
    figtitle += ' (t= %.3f kyr | dt= %.3f kyr)'%(simtime,simdt)
    fig.suptitle(figtitle)
    
    if storefig:
        return mplot.savefig(fig, filename, subpath)
    else:
        w = 5.*aspect_accu
        h = 12. if n>1 else 6.
        print(aspect)
        print(w)
        print(h)
        fig.set_size_inches(w, h, forward=True)
        fig.set_tight_layout(True)
        return fig
    
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
    varset  = getarg(2, None)
    varspec = getarg(3, None)
    if os.path.isfile(path):
        main(path, varset, varspec, storefig=False)
        plt.show()
    elif os.path.isdir(path):
        cnt, cntfiles = tools.countbatch(path, basename, cntrange, cntdigits=4)
        procf = lambda fn: main(fn, varset, varspec, storefig=True)
        tools.batchprocess(procf, cntfiles, verbose=True, strict=True)
    else:
        raise IOError('Path not understood.')

