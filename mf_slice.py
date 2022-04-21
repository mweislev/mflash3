# -*- coding: utf-8 -*-
import sys,os

import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as cl
import matplotlib.cm as cm
import matplotlib.patches as patches
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
version_str = 'cfslice814'
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

presets = {
   'vel': {'vars':['velx','vely','velz','vel'], 'dirs':[2,0,1], },
   'mag': {'vars':['magx','magy','magz','mag'], 'dirs':[2,0,1],
      'quivers':[None,None,None,None], 'blocks':[False,False,False,True], },
   'mhdr':{'vars':['dens','pres','temp','rlevel'], 'dirs':[2,0,1],
      'fields':['vel',None,None,None], 'blocks':[False,False,False,True], },
   'mhd':{'vars':['dens','dens','pres','temp'], 'dirs':[2,0,1],
      'fields':['vel',None,None,None], 'quivers':[None,'mag',None,None],
      'blocks':[False,False,True,True], },
   'vmhd':{'vars':['mach_s','dens','pres','temp'], 'dirs':[2,0,1],
      'fields':['vel',None,None,None], 'quivers':[None,'mag',None,None],
      'blocks':[False,False,True,True], },
   }

preset_default = presets['mhd']
stdax = [2,0,1]

quiver_res = (64,64,64)


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

ax_xy_order = {0:[1,2], 1:[0,2], 2:[0,1]}

ax_units = {'pc':pc, 'Au':au, 'km':1e+5, 'cm':1.}
usorter = dict(key=itemgetter(1), reverse=True)

def PickAxisUnit(extent, rth=.1):
   exmax = np.max(np.abs(extent))
   for axulabel, axunit in sorted(ax_units.items(), **usorter):
       if exmax > rth*axunit: break
   return axunit, axulabel
   
def axFormat(ax, axis, offset, vartitle, axulabel, show_axlabel=True, show_title=True):
   if show_axlabel:
      axustr = ' [%s]'%axulabel
      ax_x, ax_y = ax_xy_order[axis]
      ax.set(xlabel=['x','y','z'][ax_x]+axustr, ylabel=['x','y','z'][ax_y]+axustr)
   if show_title:
      slice_title =  ['x','y','z'][axis]
      slice_title += '= %.3g %s'%(offset, axulabel)
      ax.set_title(vartitle+' ('+slice_title+')')
   return ax   


#===============================================================================
# ==== SLICE FIELDLINE IMAGER ==================================================
#===============================================================================

def axVarFlSlice(ax, hdfdata, octree, flvarkey, axis=2, offset=0.,
            x_extent=None, y_extent=None, z_extent=None,
            show_axlabel=False, show_title=False, mode='quiver', **plargs):
   #
   flres = quiver_res if mode=='quiver' else res
   #
   bgparam = dict(x_extent=x_extent, y_extent=y_extent, z_extent=z_extent,
      xres=flres[0], yres=flres[1], zres=flres[2])
   slparam = dict(interpolation=grid_interpolation, axis=axis,
      wrap=True, verbose=False)

   # Read unit settings of the field name (without x/y/z)
   # (this settings will be used for both screen axis of the quiver/flow)
   dunit, dnorm, dcmap, dlabel, dulabel, dtitle, dweighvar = GetVarSettings(flvarkey)
   
   # Determine which variables to read for the fields axis components
   flvar = flvarkey.split('::')[0]
   xyzvar = [flvar+a for a in ['x','y','z']]
   ax_x, ax_y = ax_xy_order[axis]
   xsvar = xyzvar[ax_x]
   ysvar = xyzvar[ax_y]

   # Read screen-x-direction vector component
   with mflash.datagrid(hdfdata[xsvar], octree, ng=0, verbose=True) as dgrid_xs:
      XYZ     = dgrid_xs.get_slice_coords(axis, offset=0., **bgparam)
      data_xs = dgrid_xs.eval(*XYZ, **slparam)

   # Read screen-y-direction vector component
   with mflash.datagrid(hdfdata[ysvar], octree, ng=0, verbose=True) as dgrid_ys:
      data_ys = dgrid_ys.eval(*XYZ, **slparam)
         
   # Read simulation domain boundary coordinates
   extent_cgs = octree.extent(axis)
   if (x_extent, y_extent, z_extent)[ax_x] is not None:
      extent_cgs[0] = np.array((x_extent, y_extent, z_extent))[ax_x]
   if (x_extent, y_extent, z_extent)[ax_y] is not None:
      extent_cgs[1] = np.array((x_extent, y_extent, z_extent))[ax_y]

   # Determine display unit for coordinate axes
   axunit, axulabel = PickAxisUnit(extent_cgs, rth=.1)
   extent = extent_cgs / axunit
   
   # Determine display unit point axes coordinates of slice
   xs = XYZ[ax_x] / axunit
   ys = XYZ[ax_y] / axunit
   
   #
   if mode=='stream':
      # Normalize screen-axis vectors
      xysdata = np.stack((data_xs, data_ys))
      nv = np.linalg.norm(xysdata, axis=0)
      nx, ny = data_xs/nv, data_ys/nv
      # Construct streamplot
      strargs = dict(density=0.5, arrowstyle='fancy', color='black')
      strargs.update(plargs)
      pl = ax.streamplot(xs.T, ys.T, nx.T, ny.T, **strargs)
   elif mode=='quiver':
      qvargs = dict(color='black')
      qvargs.update(plargs)
      U = data_xs / dunit
      V = data_ys / dunit
      pl = ax.quiver(xs.T, ys.T, U.T, V.T, **qvargs)
      L = np.percentile(np.hypot(U,V), 95)
      qkstr = f'{L}$\,${dulabel}'
      qk = ax.quiverkey(pl, .8, -.1, L, qkstr, labelpos='E', coordinates='axes')
   else:
      raise NotImplementedError(f'Unknown field plot mode {mode}')
   #
   vartitle = f'{flvar} field'
   axFormat(ax, axis, offset/axunit, vartitle, axulabel, show_axlabel, show_title)
   #
   return pl


#===============================================================================
# ==== SLICE IMAGER ============================================================
#===============================================================================

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
    axFormat(ax, axis, offset/axunit, vartitle, axulabel, show_axlabel, show_title)
    #
    return im, cbar
    
    
#===============================================================================
# ==== BLOCK GRID SLICE IMAGER =================================================
#===============================================================================

from matplotlib.collections import PatchCollection

def axBlockSlice(ax, hdfdata, octree, axis=2, offset=0.,
            x_extent=None, y_extent=None, z_extent=None,
            show_axlabel=False, show_title=False, **plargs):

    dens = hdfdata['dens']
    blindex = hdfdata['blockindex']
    dgrid = mflash.datagrid(blindex, octree, ng=0, verbose=True)
    blkid_slice = dgrid.read_slice(axis, offset=offset, interpolation=None,
        x_extent=x_extent, y_extent=y_extent, z_extent=z_extent,
        xres=res[0], yres=res[1], zres=res[2])
    slice_blkids = np.array(list(set(blkid_slice.ravel()))).astype(int)

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
    block_bbox = hdfdata['bounding box'] / axunit
    slice_blk_xs_extent = block_bbox[slice_blkids,ax_x,:]
    slice_blk_ys_extent = block_bbox[slice_blkids,ax_y,:]

    slb_xmin = slice_blk_xs_extent[:,0]
    slb_ymin = slice_blk_ys_extent[:,0]
    slb_xmax = slice_blk_xs_extent[:,1]
    slb_ymax = slice_blk_ys_extent[:,1]
    slb_xlen = slb_xmax-slb_xmin
    slb_ylen = slb_ymax-slb_ymin
    slb_rect = zip(slb_xmin,slb_ymin,slb_xlen,slb_ylen)

    largs = dict(linewidth=1, edgecolor='dimgray', facecolor='none', alpha=.7)
    largs.update(plargs)
    for x,y,xl,yl in slb_rect:
        rect = patches.Rectangle((x, y), xl, yl, **largs)
        ax.add_patch(rect)
    #blk_patches = [mpl.patches.Rectangle((x, y),xl,yl, **largs) for (x,y,xl,yl) in slb_rect]
    #ax.add_collection(PatchCollection(blk_patches))
    ax.set(xlim=extent[0], ylim=extent[1])
    #
    vartitle = 'Block structure'
    axFormat(ax, axis, offset/axunit, vartitle, axulabel, show_axlabel, show_title)
    #
    return None


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

def DecodePreset(key):
    if key is None:
        preset = preset_default
    elif key in presets:
        preset = presets[key]
    else:
        preset = {'vars':[key,], 'dirs':stdax, 'fields':None}

    if varspec is None:
        datakeys = preset.get('vars')
    else:
        datakeys = ['::'.join([k,varspec]) for k in preset.get('vars')]
    
    dirs = preset.get('dirs', stdax)

    nvar = len(datakeys)
    nax  = len(dirs)
    baseshape = (nax,nvar)
    
    fields = preset.get('fields', None)
    fieldkeys = np.broadcast_to(fields, baseshape)

    quivers = preset.get('quivers', None)
    quiverkeys = np.broadcast_to(quivers, baseshape)
    
    blocks = preset.get('blocks', True)
    showbl = np.broadcast_to(blocks, baseshape)
    
    return datakeys, fieldkeys, quiverkeys, showbl, dirs
    
    
def main(plotfile, key, varspec=None, storefig=False):

    datakeys, fieldkeys, quiverkeys, showbl, dirs = DecodePreset(key)
    
    filename = os.path.basename(plotfile)
        
    if storefig: # Check if file was already made (and should be skipped)
        dirname  = os.path.dirname(plotfile)
        simname  = os.path.basename(dirname)
        subpath = f'{simname}/{key}'
        outfilepath = mplot.get_savefig_filepath(filename, subpath)
        if os.path.exists(outfilepath):
            print(f'Skipping {filename}')
            return None
        
    hdfdata, octree = OpenPlotfile(plotfile)

    extparam = dict(x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
    nvar = len(datakeys)
    nax = len(dirs)
        
    if nvar==1:
        fig, ax = plt.subplots(1, nax)
        key = datakeys[0]
        LoadVar(hdfdata, octree, key)
        dgrid = mflash.datagrid(hdfdata[key], octree, ng=nguard, verbose=True)
        for ia,a in enumerate(dirs):
            plax = ax[ia]
            extparam.update(dict(axis=a, offset=point[a]))
            axVarSlice(plax, dgrid, key, **extparam)
            axBlockSlice(plax, hdfdata, octree, **extparam)
        xlim, ylim = ax[0].get_xlim(), ax[0].get_ylim()
    else:
        fig, ax = plt.subplots(nax, nvar)
        for iv,key in enumerate(datakeys):
            LoadVar(hdfdata, octree, key)
            dgrid = mflash.datagrid(hdfdata[key], octree, ng=nguard, verbose=True)
            for ia,a in enumerate(dirs):
                plax = ax[ia,iv]
                extparam.update(dict(axis=a, offset=point[a]))
                axVarSlice(plax, dgrid, key, **extparam)
                if showbl[ia,iv]:
                    axBlockSlice(plax, hdfdata, octree, **extparam)
                field = fieldkeys[ia,iv]
                if field:
                    axVarFlSlice(plax, hdfdata, octree, field, mode='stream',
                        **extparam)
                quiver = quiverkeys[ia,iv]                
                if quiver:
                    axVarFlSlice(plax, hdfdata, octree, quiver, mode='quiver',
                        **extparam)
        xlim, ylim = ax[0,0].get_xlim(), ax[0,0].get_ylim()

    nstep = hdfdata['integer scalars/nstep']
    simtime = hdfdata['real scalars/time'] * (1e+3/Myr)
    simdt = hdfdata['real scalars/dt'] * (1e+3/Myr)
    figtitle =  '%s (step %i)'%(filename,nstep)
    figtitle += ' (t= %.3f kyr | dt= %.3f kyr)'%(simtime,simdt)
    fig.suptitle(figtitle)
    
    if storefig:
        return mplot.savefig(fig, filename, subpath)
    else:
        aspect = (xlim[1]-xlim[0]) / (ylim[1]-ylim[0])
        sqrta = np.sqrt(aspect)
        w = 5.*sqrta*nvar if nvar>1 else 22.*sqrta
        h = 12./sqrta if nvar>1 else 6./sqrta
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
    preset  = getarg(2, None)
    varspec = getarg(3, None)
    if os.path.isfile(path):
        main(path, preset, varspec, storefig=False)
        plt.show()
    elif os.path.isdir(path):
        cnt, cntfiles = tools.countbatch(path, basename, cntrange, cntdigits=4)
        procf = lambda fn: main(fn, preset, varspec, storefig=True)
        tools.batchprocess(procf, cntfiles, verbose=True, strict=True)
    else:
        raise IOError('Path not understood.')

