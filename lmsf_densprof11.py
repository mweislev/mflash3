# -*- coding: utf-8 -*-
import sys,os

import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.colors as cl
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize, PowerNorm, SymLogNorm
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.spatial import KDTree

import libmf.micflash as mflash
import libmf.micplot as mplot
import libmf.tools as tools
from var_settings_v3 import var_settings, intvar, GetVarSettings
from constants import *


#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

KS_var = 'dens::core'
version_str = 'densprof011Rtest2'

# ==== FILE SETTINGS ===========================================================
var_ch = mflash.var_ch5
basename = '*hdf5_plt*_'
#basename = '*hdf5_chk*_'
cntrange = list(range(0,500))

# ==== FIGURE SETTINGS =========================================================
fs = mplot.set_fig_preset('display')
mplot.fig_outdir = 'densprof/%s-%s'%(version_str, mplot.fig_target)
time_cmap = 'jet'

# ==== KERNEL SMOOTHER SETTINGS ================================================
N_x = 500
KS_config = dict(kernel='Epanechnikov', neigh_min=100, rels=0.05)


#===============================================================================
# ==== PLOTFILE HELPER =========================================================
#===============================================================================

def OpenPlotfile(plotfile):
   try:
      hdfdata = mflash.plotfile(plotfile)
   except:
      raise IOError()
   try:
      hdfdata.learn(mflash.var_mhd)
      hdfdata.learn(mflash.var_grid)
      hdfdata.learn(mflash.var_ch5)
   except:
      hdfdata.close()
      raise IOError()
   try:
      octree = mflash.pm3dgrid(hdfdata)
   except:
      hdfdata.close()
      raise IOError()
   return hdfdata, octree
   
def ReadSimtime(plotfile):
   with mflash.plotfile(plotfile) as hdfdata:
      simtime = hdfdata['real scalars/time']
   return simtime
   
def CenterRadii(hdfdata, octree, center=None, weights=None):
   coords = octree.coords()
   X = coords[:,0,:,:,:]
   Y = coords[:,1,:,:,:]
   Z = coords[:,2,:,:,:]
   if center is None:
      if weights is None:
         weights = hdfdata['mass']
      Xc = np.average(X, weights=weights)
      Yc = np.average(Y, weights=weights)
      Zc = np.average(Z, weights=weights)
   else:
      Xc, Yc, Zc = center
   # Get coordinates relative to center of mass
   Xrel = coords[:,0,:,:,:]-Xc
   Yrel = coords[:,1,:,:,:]-Yc
   Zrel = coords[:,2,:,:,:]-Zc
   Rrel = np.sqrt(Xrel**2+Yrel**2+Zrel**2)
   #Rrel = np.sqrt(X**2+Y**2+Z**2) #TODO XXX REMOVE AFTER TESTING XXX
   return Rrel
   
def FindPeak(hdfdata, octree, varkey='dens', direction='up'):
   coords = octree.coords()
   X = coords[:,0,:,:,:]
   Y = coords[:,1,:,:,:]
   Z = coords[:,2,:,:,:]
   vardata = hdfdata[varkey]
   if direction == 'down':
      i_flat = np.argmin(vardata)
   else:
      i_flat = np.argmax(vardata)
   I = np.unravel_index(i_flat, vardata.shape)
   peak_position = np.array([X[I], Y[I], Z[I]])
   return peak_position


#===============================================================================
# ==== SMOOTHING KERNEL ========================================================
#===============================================================================

def KernelEpanechnikov(u):
   usq = u**2
   support = usq < 1.
   value = .75*(1.-usq)
   return support*value

def KernelGaussian(u):
   usq = u**2
   support = usq < 1.
   value = 1./np.sqrt(2.*np.pi)*np.exp(-.5*usq)
   return support*value

def KernelUniform(u):
   usq = u**2
   support = usq < 1.
   value = 1.
   return support*value
   
KernelFunctions = {
   'Epanechnikov':KernelEpanechnikov,
   'Gaussian':KernelGaussian,
   'Uniform':KernelUniform,
   }


#===============================================================================
# ==== PLOT HELPER =============================================================  
#===============================================================================

def AxVarFormat(ax, varkey):
   settings = GetVarSettings(varkey)
   units, varnorm, hist_cmap, varlabel, ulabel, vartitle, weighvar = settings
   if isinstance(varnorm, LogNorm):
      ax.set_yscale('log')
   else:
      ax.set_yscale('linear')
   ax.set_ylim([varnorm.vmin, varnorm.vmax])
   ax.set_ylabel('%s %s [%s]'%(vartitle,varlabel,ulabel))
   return settings

def VarUnit(varkey):
   units, varnorm, hist_cmap, varlabel, ulabel, vartitle, weighvar = GetVarSettings(varkey)
   return units
   
   
#===============================================================================
# ==== KERNEL SMOOTHER =========================================================
#===============================================================================

def KernelSmoothedF(Xdata, Ydata, weights):
   Xd = Xdata.flatten()#np.ravel(Xdata)
   Yd = Ydata.flatten()#np.ravel(Ydata)
   Wd = weights.flatten()#np.ravel(weights)
   try:
      Tree = KDTree(Xd[:,None])
   except:
      reclim = sys.getrecursionlimit()
      print(f'WARNING: KDTree recursion limit ({reclim}) breached. Trying to increase limit.')
      sys.setrecursionlimit(2*reclim)
      Tree = KDTree(Xd[:,None])
   def SmoothedFunc(Xin, kernel='Epanechnikov', neigh_min=100, rels=0., abss=None):
      if not kernel in KernelFunctions:
         raise KeyError(f'Unknown smoothing kernel: {kernel}')
      X = np.ravel(Xin)
      # Get the next neighbors (indicies ii) for each of the sampling points X:
      foo, ii = Tree.query(X[:,None], neigh_min)
      # Calculate an symmetric interval [RX-S_neigh,RX+S+neigh] around each RX,
      # so that it exactly cover those neighbours
      Xii = Xd[ii]
      Xmin = np.min(Xii, axis=1)
      Xmax = np.max(Xii, axis=1)
      S_neigh = np.maximum(Xmax-X, X-Xmin)
      # Extend the interval by independently assuming a window S_rel,
      # that covers a range relative to X (sm_Srel*X)
      S_rel = rels*X
      # Extend the interval by local resolution, if user desires so
      if abss is not None:
         S_res = abss
      else:
         S_res = 0.        
      ##
      S = np.sqrt(S_rel**2+S_neigh**2+S_res**2)
      # EXPAND SELECTION UPON EXPANDED S ()!
      # As the number of data points in the neighborhood of each sampling point
      # is not constant, the outer index dimension of the index field I_
      # is now in form of a list.
      I_ = [Tree.query_ball_point(x,s) for (x,s) in zip(X[:,None],S)]
      # Select data inside the radius S around Xin
      X_ = [Xd[I] for I in I_]
      Y_ = [Yd[I] for I in I_]
      W_ = [Wd[I] for I in I_]
      # Apply weighted smoothing kernel
      KernelF = KernelFunctions.get(kernel, KernelEpanechnikov)
      U_ = [(x-xd)/s for (x,xd,s) in zip(X_,X[:,None],S)]
      K_ = map(KernelF, U_)
      KW_ = [K*W for (K,W) in zip(K_,W_)]
      YKW_ = [Y*KW for (Y,KW) in zip(Y_,KW_)]
      Yout = np.array(list(map(np.sum,YKW_))) / np.array(list(map(np.sum,KW_)))
      return Yout
   return SmoothedFunc


#===============================================================================
# ==== XXX =====================================================================  
#===============================================================================

def VarProfileF(hdfdata, octree, varkey, center=None, mask=None):
   vardata = hdfdata[varkey]
   if mask is None:
      mask = np.ones(vardata.shape, dtype=bool)
   if center is not None:
      Rrel = CenterRadii(hdfdata, octree, center=center)
   else:
      mass = hdfdata['mass']
      Rrel = CenterRadii(hdfdata, octree, weights=mass*mask)
   weighvar = GetVarSettings(varkey)[6]
   weights = hdfdata[weighvar]*mask
   F_smoothed = KernelSmoothedF(Rrel, vardata, weights)
   return F_smoothed

def AxVarRadScatter(ax, hdfdata, octree, varkey, **scargs):
   varpeak = 1e+9*np.array([1.,1.,1.])# FindPeak(hdfdata, octree, 'dens', 'up')
   #
   R = np.ravel(CenterRadii(hdfdata, octree, center=varpeak))
   D = np.ravel(hdfdata[varkey])
   Yu = VarUnit(varkey)
   sc = ax.scatter(R/au, D/Yu, **scargs)
   ax.set(xscale='log', xlabel='Radius [AU]')
   return sc

def AxVarRadProfile(ax, hdfdata, octree, varkey, **plargs):
   varpeak = 1e+9*np.array([1.,1.,1.])# FindPeak(hdfdata, octree, 'dens', 'up')
   # Create sampling points RX for kernel smoother
   R = np.ravel(CenterRadii(hdfdata, octree, center=varpeak))
   neigh_min = KS_config['neigh_min']
   rels = KS_config['rels']
   Rs = np.sort(R)
   RXmin = Rs[neigh_min//3]
   RXmax = Rs.max()*(1.-rels)
   RX = np.linspace(np.sqrt(RXmin), np.sqrt(RXmax), N_x) **2
   # Retrieve RY(RX) from kernel smoother
   F_var = VarProfileF(hdfdata, octree, varkey, center=varpeak)
   abss = hdfdata['lenx'].min()
   RY = F_var(RX, abss=abss, **KS_config)
   # Plot
   Yu = VarUnit(varkey)
   pl = ax.plot(RX/au, RY/Yu, **plargs)
   ax.set(xscale='log', xlabel='Radius [AU]')
   return pl


#===============================================================================
# ==== MAIN ====================================================================  
#===============================================================================

def Main(plotfile, varkey, ax_combine=None, **param):
   print(f'-> Processing {plotfile}')
   sys.stdout.flush()
   hdfdata, octree = OpenPlotfile(plotfile)
   
   plparam = dict()
   if ax_combine is not None:
      plparam.update(param)
      AxVarRadProfile(ax_combine, hdfdata, octree, varkey, **plparam)

   fig, ax = plt.subplots(1,1)
   mplot.ax_title(ax, plotfile)
   AxVarRadScatter(ax, hdfdata, octree, varkey, marker='.', color='grey', alpha=.3)
   AxVarRadProfile(ax, hdfdata, octree, varkey, color='black')
   AxVarFormat(ax, varkey)

   dirname  = os.path.dirname(plotfile)
   filename = os.path.basename(plotfile)
   simname  = os.path.basename(dirname)
   mplot.savefig(fig, filename, simname)
   
   hdfdata.close()
   
def ProcessDir(dirname, dirfiles, varkey):
   if not dirfiles:
      return
   plotfiles = [os.path.join(dirname, filename) for filename in dirfiles]
   # Get color from time of each plotfile
   pf_simtime = [ReadSimtime(pf) for pf in plotfiles]
   t_min = np.min(pf_simtime)
   t_max = np.max(pf_simtime)
   t_norm = Normalize(t_min, t_max)
   t_cmap = get_cmap(time_cmap)
   pf_color = [t_cmap(t_norm(t)) for t in pf_simtime]
   #
   figD, axD = plt.subplots(1,1)
   # Process plotfiles
   for i, pf in enumerate(plotfiles):
      color = pf_color[i]
      Main(pf, varkey, ax_combine=axD, color=color)
   # Format plot
   mplot.ax_title(axD, dirname)
   AxVarFormat(axD, varkey)
   # Add colorbar for time dependent linecolor
   divider = make_axes_locatable(axD)
   c1ax = divider.append_axes('right', size=0.3, pad=0.1)
   cb1 = mpl.colorbar.ColorbarBase(c1ax, cmap=t_cmap, norm=t_norm,
      orientation='vertical')
   cb1.set_label('Time [s]')
   # Save plot
   dirname_ = os.path.dirname(pf) # This hack removes an occasional trailing backslash
   simname = os.path.basename(dirname_)
   mplot.savefig(figD, 'total', subpath=simname)
            

#===============================================================================
# ==== LOBBY ===================================================================
#===============================================================================

def getarg(i, default):
    try:
        return sys.argv[i]
    except:
        return default
        
if __name__ == '__main__':
   path = sys.argv[1]
   if os.path.isfile(path):
      Main(path, KS_var)
   elif os.path.isdir(path):
      dirdict = tools.dirbatch(path, basename, cntrange, cntdigits=4)
      for dirname, dirfiles in dirdict.items():
         ProcessDir(dirname, dirfiles, KS_var)
   else:
      raise IOError('Path not understood.')
   print('=== FINISHED! ===')

