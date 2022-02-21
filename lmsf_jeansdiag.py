# -*- coding: utf-8 -*-
import sys,os

import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as cl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import libmf.micflash as mflash
import libmf.micplot as mplot
import libmf.tools as tools
from var_settings_v3 import var_settings, intvar, GetVarSettings

from constants import *


#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

# ==== FILE SETTINGS ===========================================================
var_ch = mflash.var_ch5
basename = '*hdf5*_'
cntrange = list(range(0,500))


#===============================================================================
# ==== HELPER ==================================================================
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
    

#===============================================================================
# ==== XXX =====================================================================  
#===============================================================================

def DiagJRef(plotfile):
   hdfdata, octree = OpenPlotfile(plotfile)
   print('Plotfile: %s'%plotfile)
   
   coords = octree.coords()
   X = coords[:,0,:,:,:]
   Y = coords[:,1,:,:,:]
   Z = coords[:,2,:,:,:]
   
   mass = hdfdata['mass']
   XcM = np.average(X, weights=mass)
   YcM = np.average(Y, weights=mass)
   ZcM = np.average(Z, weights=mass)
   print('CtrM coords       %.5g  %.5g  %.5g'%(XcM,YcM,ZcM))

   dens = hdfdata['dens']

   P_f = np.argmax(dens)
   P = np.unravel_index(P_f, dens.shape)

   res = hdfdata['length'][P]
   l_jeans = hdfdata['l_jeans'][P]
   rlevel = hdfdata['rlevel'][P]
   
   print('Peak coords       %.5g  %.5g  %.5g'%(X[P],Y[P],Z[P]))
   print('Peak density:     %.5g'%dens[P])
   print('Peak l_jeans:     %.5g'%l_jeans)
   print('Peak resolution:  %.5g'%res)
   print('Peak rlevel:      %.5g'%rlevel)
   print('Peak rl_jreq:     %.5g'%hdfdata['jref_ltarget'][P])
   print()
   

#===============================================================================
# ==== MAIN ====================================================================  
#===============================================================================

def Main(plotfile):
   DiagJRef(plotfile)


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
      Main(path)
   elif os.path.isdir(path):
      cnt, cntfiles = tools.countbatch(path, basename, cntrange, cntdigits=4)
      procf = lambda fn: Main(fn)
      tools.batchprocess(procf, cntfiles, verbose=False, strict=True)
   else:
      raise IOError('Path not understood.')

