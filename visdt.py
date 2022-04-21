import sys,os
import re
import numpy as np
import matplotlib.pylab as plt
import libmf.tools as tools
from matplotlib.colors import LogNorm, Normalize, PowerNorm, SymLogNorm
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

#===============================================================================
# ==== CONSTANTS ===============================================================
#===============================================================================

from constants import *


#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

stepfile_pattern = 'slurm*.out'
nguard = 4
nxb = 8
ax_xy_order = {0:[1,2], 1:[0,2], 2:[0,1]}
dtq_norm = LogNorm(1e-2, 1e+2)
dtq_cmap = get_cmap('RdYlBu')


#===============================================================================
# ==== XXX =====================================================================
#===============================================================================

stepline_signals = ('  ', 'E+', '(', ', ', ') |  ')

def IsStepLine(line):
    is_stepline = True
    for s in stepline_signals:
        if not s in line:
            is_stepline = False
            break
    return is_stepline
    
def ReadStepLine(line):
    text = re.sub('[|(),]', '', line)
    words = text.split()
    values = list(map(tools.safe_eval, words))
    if not len(values) == 7:
        raise RuntimeError('XXX')
    return values

def ReadStepfile(filepath):
    with open(filepath, 'r') as infile:
        linetable = list()
        for il,line in enumerate(infile):
            if not IsStepLine(line):
                continue
            else:
                try:
                    linevalues = ReadStepLine(line)
                except RuntimeError:
                    continue
                else:
                    linetable.append(linevalues)
        return linetable


#===============================================================================
# ==== SIMSTEPS ================================================================  
#===============================================================================

def ShowSimSteps(simpath, dirfiles):
    linetable = list()
    for fn in dirfiles:
        filepath = os.path.join(simpath,fn)
        linetable += ReadStepfile(filepath)

    cols = zip(*linetable)
    colarrays = map(np.array, cols)
    nstep, time, dt, xdt, ydt, zdt, dt_hydro = colarrays
    
    parfp = os.path.join(simpath, 'flash.par')
    pardict = tools.readpar(parfp)

    xmin = pardict['xmin']
    xmax = pardict['xmax']
    ymin = pardict['ymin']
    ymax = pardict['ymax']
    zmin = pardict['zmin']
    zmax = pardict['zmax']
    lrefine_min = pardict['lrefine_min']
    rootbl_extent = (xmax-xmin) / pardict['nblockx']
    cellsize_max = rootbl_extent / (nxb*2**int(lrefine_min-1))
    guardsize_max = nguard*cellsize_max
    
    xmin_gc = xmin -guardsize_max
    xmax_gc = xmax +guardsize_max
    ymin_gc = ymin -guardsize_max
    ymax_gc = ymax +guardsize_max
    zmin_gc = zmin -guardsize_max
    zmax_gc = zmax +guardsize_max
    
    tmin = np.min(time)
    tmax = np.max(time)
    
    nmin = np.min(nstep)
    nmax = np.max(nstep)
    dteff = (tmax-tmin)/(nmax-nmin)
    
    figures = list()
    
    fig1, axs = plt.subplots(2, sharex=True)
    ax0, ax1 = axs
    
    tpstyle = dict(alpha=.5, ls='', marker='.')
    ax0.plot(time, xdt, color='red', label='X', **tpstyle)
    ax0.plot(time, ydt, color='green', label='Y', **tpstyle)
    ax0.plot(time, zdt, color='blue', label='Z', **tpstyle)
    ax0.plot([tmin,tmax], [xmin_gc,xmin_gc], color='black', ls='-')
    ax0.plot([tmin,tmax], [xmax_gc,xmax_gc], color='black', ls='-')
    ax0.plot([tmin,tmax], [xmin,xmin], color='black', ls=':')
    ax0.plot([tmin,tmax], [xmax,xmax], color='black', ls=':')
    ax0.set(xlabel='$t$')
    ax0.set(ylabel='Pos(d$t_\mathrm{min}$)')
    ax0.legend(loc='upper right')

    ttstyle = dict(alpha=.7)
    ax1.plot(time, dt, label='d$t$', color='teal', **ttstyle)
    ax1.plot(time, dt, label='d$t_\mathrm{hydro}$', color='indigo', **ttstyle)
    ax1.plot([tmin,tmax], [dteff,dteff], label='<d$t$>', color='black', linestyle=':')
    ax1.set(xlabel='$t$')
    ax1.set(yscale='log', ylabel='d$t$')
    ax1.legend(loc='upper right')
    
    dirname  = os.path.dirname(simpath) # This hack removes a possible trailing '\'
    simname  = os.path.basename(dirname)
    fig1.suptitle(simname)

    figures.append(fig1)
    
    
    for sax in [0,1,2]:
        fig, ax = plt.subplots()
        ax_x, ax_y = ax_xy_order[sax]
        Xs = [xdt,ydt,zdt][ax_x]
        Ys = [xdt,ydt,zdt][ax_y]
        dtq = dt/dteff
        C = dtq_cmap(dtq_norm(dtq))
        sc = ax.scatter(Xs, Ys, color=C, marker='x')

        divider = make_axes_locatable(ax)
        c1ax = divider.append_axes("right", size=0.3, pad=0.1)
        cb1 = mpl.colorbar.ColorbarBase(c1ax, cmap=dtq_cmap, norm=dtq_norm, orientation='vertical')
        cb1.set_label('d$t$ / <d$t$>')

        Xsmin = [xmin,ymin,zmin][ax_x]
        Xsmax = [xmax,ymax,zmax][ax_x]
        Ysmin = [xmin,ymin,zmin][ax_y]
        Ysmax = [xmax,ymax,zmax][ax_y]
        ax.plot([Xsmin, Xsmin], [Ysmin, Ysmax], color='black')
        ax.plot([Xsmin, Xsmax], [Ysmax, Ysmax], color='black')
        ax.plot([Xsmax, Xsmax], [Ysmax, Ysmin], color='black')
        ax.plot([Xsmax, Xsmin], [Ysmin, Ysmin], color='black')

        Xsmin_gc = [xmin_gc,ymin_gc,zmin_gc][ax_x]
        Xsmax_gc = [xmax_gc,ymax_gc,zmax_gc][ax_x]
        Ysmin_gc = [xmin_gc,ymin_gc,zmin_gc][ax_y]
        Ysmax_gc = [xmax_gc,ymax_gc,zmax_gc][ax_y]
        ax.plot([Xsmin_gc, Xsmin_gc], [Ysmin_gc, Ysmax_gc], color='black', ls=':')
        ax.plot([Xsmin_gc, Xsmax_gc], [Ysmax_gc, Ysmax_gc], color='black', ls=':')
        ax.plot([Xsmax_gc, Xsmax_gc], [Ysmax_gc, Ysmin_gc], color='black', ls=':')
        ax.plot([Xsmax_gc, Xsmin_gc], [Ysmin_gc, Ysmin_gc], color='black', ls=':')
        
        ax.set(xlabel=['X','Y','Z'][ax_x])
        ax.set(ylabel=['X','Y','Z'][ax_y])
        
        fig.suptitle(simname)
        figures.append(fig)
        
    
    return figures


#===============================================================================
# ==== MAIN ====================================================================  
#===============================================================================

def main(path):
    stepfiles_dirdict = tools.scanpath(path, stepfile_pattern)
    for simpath,dirfiles in stepfiles_dirdict.items():
        ShowSimSteps(simpath, dirfiles)
    plt.show()


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
    main(path)

