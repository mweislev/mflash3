# -*- coding: utf-8 -*-
from matplotlib.colors import LogNorm, Normalize, PowerNorm, SymLogNorm
from matplotlib.cm import get_cmap
import libmf.micplot as mplot
import numpy as np

__author__ = "Michael Weis"
__version__ = "1.0.0.0"

# ==== PHYSICS =================================================================
from constants import *

# ==== CUSTOM COLORMAPS ========================================================
mplot.build_shelixmap('mcwA', start=.88, rotations=2./3., sat=.75, gamma=1.0, Ymin=0.001, Ymax=0.999)
mplot.build_shelixmap('mcwB', start=.88, rotations=2./3., sat=.75, gamma=1.0, Ymin=0.001, Ymax=0.92)
mplot.build_shelixmap('mcwC', start=.88, rotations=2./3., sat=.75, gamma=1.0, Ymin=0.01, Ymax=0.67)

# ==== SETTINGS ================================================================
intvar = ['rlevel',]

var_settings = { # units, norm, cmap, label, title, column weighting
    'dens::corelow':[1., LogNorm(1.0e-24, 1.0e-14), get_cmap('CMRmap_r'),
        r'$\rho$', 'g/cm$^{3}$', 'Density', 'vol'],
    'dens::corehigh':[1., LogNorm(3.16e-19, 3.16e-12), get_cmap('CMRmap_r'),
        r'$\rho$', 'g/cm$^{3}$', 'Density', 'vol'],

    'dens::core':[1., LogNorm(1.0e-22, 1.0e-12), get_cmap('CMRmap_r'),
        r'$\rho$', 'g/cm$^{3}$', 'Density', 'vol'],
    'pres::core':[1., LogNorm(1.0e-10, 1.0e-6), get_cmap('CMRmap_r'),
        r'$P_\mathrm{therm}$', 'Ba', 'Thermal Pressure', 'vol'],

    'ents': [1., LogNorm(40., 65.), get_cmap('CMRmap'),
        r'$s$', 'dimless', 'Entropy per Particle', 'vol'],
    'entdens': [-1.*m_a, LogNorm(1., 1e+6), get_cmap('CMRmap'),
        r'$S(\rho)$', 'dimless', 'Physical Entropy Density', 'vol'],
    'entropy': [-1.*m_a*pc**3, LogNorm(1e-10, 1e+20), get_cmap('CMRmap'),
        r'$\int S\,\mathrm{d}\,V$', 'dimless', 'Integrated Entropy', None],


    'dens::rotor':[1., LogNorm(.2, 25.), get_cmap('CMRmap_r'),
        r'$\rho$', 'dimless', 'Density', 'vol'],
    'pres::rotor':[1., LogNorm(.02, 2.5), get_cmap('CMRmap_r'),
        r'$P$', 'dimless', 'Pressure', 'vol'],
    'magp_2d::rotor':[1., LogNorm(.01, 0.5), get_cmap('CMRmap_r'),
        r'$P_\mathrm{mag}$', 'dimless', 'Magnetic Pressure', 'vol'],
    'mach_s_2d::rotor':[1., LogNorm(.01, 100.), get_cmap('CMRmap_r'),
        r'$c_\mathrm{s}$', 'Mach', 'Mach Number', 'vol'],

    'dens::torus':[1., LogNorm(1e-4, 1.), get_cmap('CMRmap_r'),
        r'$\rho$', 'dimless', 'Density', 'vol'],

    'numdens::tile': [1., LogNorm(.7, 1.4e+7), get_cmap('CMRmap_r'),
        r'$n$', 'parts/cm$^{3}$', 'Particle Number Density', 'vol'],
    'numdens::mov': [1., LogNorm(.7, 1.4e+4), get_cmap('mcwC_r'),
        r'$n$', 'parts/cm$^{3}$', 'Particle Number Density', 'vol'],
    'velx::mov': [km, Normalize(-5., 5.), get_cmap('bwr'),
        r'$v_{x}$', 'km/s', 'X-Velocity', 'mass'],
    

    'c_s': [km, Normalize(0., 7.5), get_cmap('CMRmap'),
        r'$c_{s}$', 'km/s', 'Wave Propagation Speed', 'mass'],
    'c_s::log': [km, LogNorm(0.1, 10.), get_cmap('CMRmap'),
        r'$c_{s}$', 'km/s', 'Wave Propagation Speed', 'mass'],

    'lcdi::log': [1., LogNorm(0.1, 10.), get_cmap('seismic'),
        r'$\sigma_{3d}/c_{s}$', 'Mach', 'Local Velocity Dispersion', 'mass'],
    'lcdi::pwr': [1., PowerNorm(.5, 0., 10.), get_cmap('Spectral_r'),
        r'$\sigma_{3d}/c_{s}$', 'Mach', 'Local Velocity Dispersion', 'mass'],
    'lcdi': [1., LogNorm(.05, 20.), get_cmap('Spectral_r'),
        r'$\sigma_{3d}/c_{s}$', 'Mach', 'Local Velocity Dispersion', 'mass'],

    'lvdi': [km, Normalize(0., 7.5), get_cmap('Spectral_r'),
        r'$\sigma_{3d}$', 'km/s', 'Local Velocity Dispersion', 'mass'],

    'lvdi_sq': [km**2, Normalize(0., 81.), get_cmap('Spectral_r'),
        r'$\sigma_{3d}^2$', 'km$^{2}$s$^{-2}$', 'Local Sq. Velocity Dispersion', 'mass'],

    'cellcount': [1., LogNorm(1.,1e+8), get_cmap('YlGnBu'),
        r'$n_\mathrm{cell}$', 'count', 'Number of cells', 'cellcount'],

    'blnd': [1., Normalize(-0.0001,1.0001), get_cmap('jet'),
        r'$\alpha_{\mathrm{DG}}$', 'Fraction', 'DGFV blending', 'cellcount'],
    'blnd::m': [1., Normalize(-0.0001,1.0001), get_cmap('jet'),
        r'$\alpha_{\mathrm{DG}}$', 'Fraction', 'DGFV blending', 'mass'],
        
    'dens::t': [1., LogNorm(1e-21, 1e-19), get_cmap('Spectral_r'),
        r'$\rho$', 'g/cm$^{3}$', 'Density', 'vol'],
    'dih2::pos': [1./Myr, Normalize(0.0,2.5), get_cmap('YlGnBu'),
        r'$DIH2$', 'Fraction/Myr', 'H2 Formation Rate', 'mass'],
    'dih2::neg': [-1./Myr, Normalize(0.0,2.5), get_cmap('YlOrBr'),
        r'$DIH2$', 'Fraction/Myr', 'H2 Destruction Rate', 'mass'],
    'dih2::avgtile': [1./Myr, Normalize(-0.3,0.3), get_cmap('bwr_r'),
        r'$DIH2$', 'Fraction/Myr', 'H2 Conversion Rate', 'mass'],

    'dens::h2tile': [1., LogNorm(1e-24, 1e-19), get_cmap('Spectral_r'),
        r'$\rho$', 'g/cm$^{3}$', 'Density', 'mass_h2'],
    'dens::cotile': [1., LogNorm(1e-24, 1e-19), get_cmap('Spectral_r'),
        r'$\rho$', 'g/cm$^{3}$', 'Density', 'mass_co'],

    'dens::tile': [1., LogNorm(1e-24, 1e-19), get_cmap('Spectral_r'),
        r'$\rho$', 'g/cm$^{3}$', 'Density', 'vol'],
    'cdto::tile': [1., LogNorm(.1,10.), get_cmap('CMRmap_r'),
        r'$A_{v}$', 'mag', 'Visual Extinction', 'vol'],

    'dens::p1histo': [1., LogNorm(1e-24, 1e-16), get_cmap('Spectral_r'),
        r'$\rho$', 'g$\,$cm$^{-3}$', 'Density', 'vol'],
    'dens::p2histo': [1., LogNorm(1e-25, 3.17e-20), get_cmap('Spectral_r'),
        r'$\rho$', 'g$\,$cm$^{-3}$', 'Density', 'vol'],
    'temp::p1histo': [1., LogNorm(8., 1.25e+4), get_cmap('Spectral'),
        r'$T$', 'K', 'Temperature', 'parts'],
    'temp::p2histo': [1., LogNorm(3.17, 3.17e+4), get_cmap('Spectral'),
        r'$T$', 'K', 'Temperature', 'parts'],
    'cdto::p1histo': [1., Normalize(0., 6.), get_cmap('CMRmap_r'),
        r'$A_{v}$', 'mag', 'Visual Extinction', 'vol'],
    'cdto::log': [1., LogNorm(.1,10.**2.5), get_cmap('CMRmap_r'),
        r'$A_{v}$', 'mag', 'Visual Extinction', 'vol'],

    'cdto::efflog': [1., LogNorm(.5,10.**2), get_cmap('CMRmap_r'),
        r'$A_{v}$', 'mag', 'Visual Extinction', 'vol'],
    'cdto::effhisto': [1., Normalize(0., 40.), get_cmap('CMRmap_r'),
        r'$A_\mathrm{v,3D}$', 'mag', 'Visual Extinction', 'vol'],
    'cdto::koenives': [1., Normalize(0., 60.), get_cmap('CMRmap_r'),
        r'$A_{v}$', 'mag', 'Visual Extinction', 'vol'],

    'abco_dens::switch': [m_a, LogNorm(1e-10, 1.), get_cmap('skinhelix_r'),
        r'$\rho$', 'u/cm$^{3}$', 'Density', 'vol'],
    'abco_dens::colpaper': [m_a, LogNorm(1., 1e+4), get_cmap('skinhelix_r'),
        r'$\rho$', 'u/cm$^{3}$', 'Density', 'vol'],
    'dens::colpaper': [m_a, LogNorm(1., 1e+4), get_cmap('skinhelix_r'),
        r'$\rho$', 'u/cm$^{3}$', 'Density', 'vol'],
    'dens::colpaperx': [1./(128.*pc), LogNorm(4.e-4, 1.e+0), get_cmap('skinhelix_r'),
        r'$\Sigma$', 'g/cm$^{2}$', 'Column Density (X)', 'vol'],
    'dens::colpapery': [1./(32.*pc), LogNorm(1.e-4, 1.e+0), get_cmap('skinhelix_r'),
        r'$\Sigma$', 'g/cm$^{2}$', 'Column Density (Y)', 'vol'],
    'dens::colpaperz': [1./(32.*pc), LogNorm(1.e-4, 1.e+0), get_cmap('skinhelix_r'),
        r'$\Sigma$', 'g/cm$^{2}$', 'Column Density (Z)', 'vol'],
#    'dens::colpaper': [m_a, LogNorm(1., 1e+4), get_cmap('skinhelix_r'),
#        r'$\rho$', 'u/cm$^{3}$', 'Density', 'vol'],
    'numdens::low': [1., LogNorm(3.16e-1, 3.16e+3), get_cmap('CMRmap_r'),
        r'$n$', 'parts/cm$^{3}$', 'Particle Number Density', 'vol'],
    'numdens': [1., LogNorm(.7, 1.4e+8), get_cmap('CMRmap_r'),
        r'$n$', 'parts/cm$^{3}$', 'Particle Number Density', 'vol'],
    'numdens::clcol': [1., LogNorm(1e+1, 1e+5), get_cmap('mcwA'),
        r'$n$', 'parts/cm$^{3}$', 'Particle Number Density', 'vol'],
    'abund_co::clcol': [1., LogNorm(1.6e-5, 1.6e-4), get_cmap('mcwB_r'),
        r'$\chi$(CO)$', 'ratio', 'CO Abundance', 'mass'],
    'numdens::col': [1., LogNorm(1., 1e+3), get_cmap('mcwA'),
        r'$n$', 'parts/cm$^{3}$', 'Particle Number Density', 'vol'],

    'n_h2::col': [1., LogNorm(1., 1e+3), get_cmap('mcwA'),
        r'$n_{H2}$', 'parts/cm$^{3}$', 'Particle Number Density', 'vol'],

    'abund_co::col': [1., LogNorm(1.5e-7, 1.5e-4), get_cmap('mcwB_r'),
        r'$\chi$(CO)$', 'ratio', 'CO Abundance', 'mass'],
    'numdens::col2': [1., LogNorm(1., 1e+3), get_cmap('CMRmap'),
        r'$n$', 'parts/cm$^{3}$', 'Particle Number Density', 'vol'],
    'abund_co::col2': [1., LogNorm(1.5e-7, 1.5e-4), get_cmap('skinhelix_r'),
        r'$\chi$(CO)$', 'ratio', 'CO Abundance', 'mass'],
    'numdens::cf99': [1., LogNorm(1., 1e+7), get_cmap('CMRmap_r'),
        r'$n$', 'parts/cm$^{3}$', 'Particle Number Density', 'vol'],
    'n_e': [1., LogNorm(1e-3, 1e+3), get_cmap('CMRmap_r'),
        r'$n_{e}$', 'parts/cm$^{3}$', 'Electron Number Density', 'vol'],
    'n_hp': [1., LogNorm(1e-6, 1e-1), get_cmap('CMRmap_r'),
        r'$n_{H_{+}}$', 'parts/cm$^{3}$', 'H$_{+}$ Particle Number Density', 'vol'],
    'n_ha': [1., LogNorm(1e-2, 1e+4), get_cmap('CMRmap_r'),
        r'$n_{H_{a}}$', 'parts/cm$^{3}$', 'H$_{a}$ Particle Number Density', 'vol'],
    'n_h2': [1., LogNorm(1e-2, 1e+4), get_cmap('CMRmap_r'),
        r'$n_{H2}$', 'parts/cm$^{3}$', 'H$_{2}$ Particle Number Density', 'vol'],
    'n_cp': [1., LogNorm(1e-4, 1e+1), get_cmap('CMRmap_r'),
        r'$n_{C+}$', 'parts/cm$^{3}$', 'C$_{+}$ Particle Number Density', 'vol'],
    'n_co': [1., LogNorm(1e-12, 1e-2), get_cmap('CMRmap_r'),
        r'$n_{CO}$', 'parts/cm$^{3}$', 'CO Particle Number Density', 'vol'],
    'nrate_h2': [1./Myr, LogNorm(1e-3, 1e+3), get_cmap('CMRmap_r'),
        r'$\Delta n_{H2}$', 'parts/cm$^{3}$/Myr', 'H$_{2}$ Particle Formation Density', 'vol'],
    'nrate_co': [1./Myr, LogNorm(1e-13, 1e-3), get_cmap('CMRmap_r'),
        r'$\Delta n_{CO}$', 'parts/cm$^{3}$/Myr', 'CO Particle Formation Density', 'vol'],
    'abund_h2': [1., LogNorm(1e-9, 1.), get_cmap('CMRmap_r'),
        r'$\chi$(H$_{2}$)', 'ratio', 'H$_{2}$ Abundance', 'mass'],
    'abund_h2::col': [1., LogNorm(1e-3, 1.), get_cmap('CMRmap'),
        r'$\chi$(H$_{2}$)', 'ratio', 'H$_{2}$ Abundance', 'mass'],
    'abund_h2::high': [1., LogNorm(1e-2, 1.), get_cmap('CMRmap'),
        r'$\chi$(H$_{2}$)', 'ratio', 'H$_{2}$ Abundance', 'mass'],
    'abund_h2::lin': [1., Normalize(0., .5), get_cmap('CMRmap'),
        r'$\chi$(H$_{2}$)', 'ratio', 'H$_{2}$ Abundance', 'mass'],
    'abund_hp': [1., LogNorm(1.5e-10, 1.5e-2), get_cmap('CMRmap_r'),
        r'$\chi$(H$_{+}$)', 'ratio', 'H$_{+}$ Abundance', 'mass'],
    'abund_co': [1., LogNorm(1e-16, 2e-4), get_cmap('CMRmap'),
        r'$\chi$(CO)', 'ratio', 'CO Abundance', 'mass'],
    'abund_co::high': [1., LogNorm(1e-5, 2e-4), get_cmap('CMRmap'),
        r'$\chi$(CO)', 'ratio', 'CO Abundance', 'mass'],
    'abund_co::lin': [1., Normalize(0., 1.5e-4), get_cmap('CMRmap'),
        r'$f_{\chi(CO)}=\frac{n_{CO}}{\rho/(m_H*ch_{abar})}$', 'ratio', 'CO Abundance', 'mass'],
    'abund_co::lh': [1., Normalize(1.25e-4, 1.5e-4), get_cmap('CMRmap'),
        r'$f_{\chi(CO)}=\frac{n_{CO}}{\rho/(m_H*ch_{abar})}$', 'ratio', 'CO Abundance', 'mass'],
        
    'abund_co::cl': [1., PowerNorm(np.log(.5)/np.log(1e-4/1.5e-4), 0., 1.5e-4), get_cmap('RdBu_r'),
        r'$f_{\chi(CO)}=\frac{n_{CO}}{\rho/(m_H*ch_{abar})}$', 'ratio', 'CO Abundance', 'mass'],
        
    'abund_cp': [1., LogNorm(1e-16, 2e-4), get_cmap('CMRmap_r'),
        r'$\chi$(C$_{+}$)', 'ratio', 'C$_{+}$ Abundance', 'mass'],
    'abund_cp::high': [1., LogNorm(1e-5, 2e-4), get_cmap('CMRmap_r'),
        r'$\chi$(C$_{+}$)', 'ratio', 'C$_{+}$ Abundance', 'mass'],
    'abund_e': [1., LogNorm(5e-5, 0.005), get_cmap('CMRmap_r'),
        r'$\chi$(e)', 'ratio', 'Electron Abundance', 'mass'],
    'dens': [1., LogNorm(1e-24, 1e-16), get_cmap('Spectral_r'),
        r'$\rho$', 'g/cm$^{3}$', 'Density', 'vol'],
    'dens::high': [1., LogNorm(3.16e-22, 3.16e-14), get_cmap('Spectral_r'),
        r'$\rho$', 'g/cm$^{3}$', 'Density', 'vol'],
    'dens::mp': [m_a, LogNorm(.7, 1.2e+7), get_cmap('Spectral_r'),
        r'$\rho$', 'u/cm$^{3}$', 'Density', 'vol'],
    'dens::col': [m_a, LogNorm(1., 1e+3), get_cmap('magma'),
        r'$\rho$', 'u/cm$^{3}$', 'Density', 'vol'],
    'dens_h1': [m_a, LogNorm(.1, 1e+4), get_cmap('CMRmap_r'),
        r'$\rho_{H1}$', 'u/cm$^{3}$', 'H$_{a}$,H$_{+}$ Density', 'vol'],
    'dens_h2': [m_a, LogNorm(1e-2, 1e+4), get_cmap('CMRmap_r'),
        r'$\rho_{H2}$', 'u/cm$^{3}$', 'H$_{2}$ Density', 'vol'],
    'dens_h2::col': [m_a, LogNorm(1e-2, 1e+4), get_cmap('CMRmap_r'),
        r'$\rho_{H2}$', 'u/cm$^{3}$', 'H$_{2}$ Density', 'vol'],
    'dens_cp': [m_a, LogNorm(2e-4, 2e+1), get_cmap('CMRmap_r'),
        r'$\rho_{C+}$', 'u/cm$^{3}$', 'C$_{+}$ Density', 'vol'],
    'dens_co': [m_a, LogNorm(1e-13, 1e-1), get_cmap('CMRmap_r'),
        r'$\rho_{CO}$', 'u/cm$^{3}$', 'CO Density', 'vol'],
    'dens_co::high': [m_a, LogNorm(1e-6, 1e+2), get_cmap('CMRmap_r'),
        r'$\rho_{CO}$', 'u/cm$^{3}$', 'CO Density', 'vol'],
    'densrate_h2': [m_a/Myr, Normalize(-10., 200.), get_cmap('CMRmap_r'),
        r'$\Delta\rho_{H2}$', 'm$_{A}$/cm$^{3}$/Myr', 'H$_{2}$ Formation Density', 'vol'],
    'densrate_h2::log': [m_a/Myr, LogNorm(1e-6, 1e+4), get_cmap('CMRmap_r'),
        r'$\Delta\rho_{H2}$', 'm$_{A}$/cm$^{3}$/Myr', 'H$_{2}$ Formation Density', 'vol'],
    'densrate_co': [m_a/Myr, LogNorm(1e-10, 1e+2), get_cmap('CMRmap_r'),
        r'$\Delta\rho_{CO}$', 'm$_{A}$/cm$^{3}$/Myr', 'CO Formation Density', 'vol'],
    'dico': [1./Myr, Normalize(-.01,.01), get_cmap('CMRmap_r'),
        r'$DICO$', 'Abund./Myr', 'CO Conversion Rate', 'mass'],
    'dih2': [1./Myr, Normalize(-0.15,0.5), get_cmap('nipy_spectral_r'),
        r'$DIH2$', 'Abund./Myr', 'H2 Conversion Rate', 'mass'],
    'relrate_h2': [1./Myr, LogNorm(1e-5, 1e+4), get_cmap('CMRmap_r'),
        r'$\frac{\Delta M_{H_{2}}}{M_{H,H^{+}}}/\Delta t$', 'fraction/Ma',
        'Proportional H$_{2}$ Formation Rate', 'mass'],
    'relrate_co': [1./Myr, LogNorm(1e-9, 1e+4), get_cmap('CMRmap_r'),
        r'$\frac{\Delta M_{CO}}{M_{C}}/\Delta t$', 'fraction/Ma',
        'Proportional CO Formation Rate', 'mass'],
    'relrate_h2::cf99': [1./Myr, LogNorm(1e-8, 2e+3), get_cmap('CMRmap_r'),
        r'$\frac{\Delta M_{H_{2}}}{M_{H,H^{+}}}/\Delta t$', 'fraction/Myr',
        'Proportional H$_{2}$ Formation Rate', 'mass'],
    'relrate_co::cf99': [1./Myr, LogNorm(1e-9, 2e+4), get_cmap('CMRmap_r'),
        r'$\frac{\Delta M_{CO}}{M_{C}}/\Delta t$', 'fraction/Myr',
        'Proportional CO Formation Rate', 'mass'],
    'Ai': [1., Normalize(1.25, 2.75), get_cmap('CMRmap_r'),
        r'$A$', 'count', 'Nucleon Number', 'parts'],
    'temp': [1., LogNorm(8., 1.25e+4), get_cmap('Spectral_r'),
        r'$T$', 'Kelvin', 'Temperature', 'parts'],
    'tdus': [1., Normalize(7.5, 17.5), get_cmap('rainbow'),
        r'$T_{dust}$', 'Kelvin', 'Dust Temperature', 'mass'],
    'pres': [1., LogNorm(1e-14, 1e-9), get_cmap('CMRmap_r'),
        r'$P_\mathrm{therm}$', 'barye', 'Thermal Pressure', 'vol'],
    'pres::k': [k_b, LogNorm(1e+2, 1e+8), get_cmap('CMRmap_r'),
        r'$P_\mathrm{therm}/k_{B}$', 'K$\,$cm$^{-3}$', 'Thermal Pressure', 'vol'],
    'magp': [1., LogNorm(1e-14, 1e-9), get_cmap('CMRmap_r'),
        r'$P_{B}$', 'barye', 'Magnetic Pressure', 'vol'],
    'magp::k': [k_b, LogNorm(1e+2, 1e+8), get_cmap('CMRmap_r'),
        r'$P_{B}/k_{B}$', 'K$\,$cm$^{-3}$', 'Magnetic Pressure', 'vol'],
    'velp': [1., LogNorm(1e-14, 1e-9), get_cmap('CMRmap_r'),
        r'$P_{ram}$', 'barye', 'Turbulent Pressure', 'vol'],
    'velp::k': [k_b, LogNorm(1e+2, 1e+8), get_cmap('CMRmap_r'),
        r'$P_{ram}/k_{B}$', 'K$\,$cm$^{-3}$', 'Turbulent Pressure', 'vol'],
    'gravp': [-1., LogNorm(1e-14, 1e-9), get_cmap('CMRmap_r'),
        r'$-P_{G}$', 'barye', 'Gravitational Pressure', 'vol'],
    'gravp::k': [-1.*k_b, LogNorm(1e+2, 1e+8), get_cmap('CMRmap_r'),
        r'$-P_{G}/k_{B}$', 'K$\,$cm$^{-3}$', 'Gravitational Pressure', 'vol'],
    'numdens::k4': [1., LogNorm(3.16e+1, 3.16e+8), get_cmap('CMRmap_r'),
        r'$n$', 'parts/cm$^{3}$', 'Particle Number Density', 'vol'],
    'pres::k4': [k_b, LogNorm(1e+1, 1e+12), get_cmap('CMRmap_r'),
        r'$P_{therm}/k_{B}$', 'K$\,$cm$^{-3}$', 'Thermal Pressure', 'vol'],
    'magp::k4': [k_b, LogNorm(1e+1, 1e+12), get_cmap('CMRmap_r'),
        r'$P_{B}/k_{B}$', 'K$\,$cm$^{-3}$', 'Magnetic Pressure', 'vol'],
    'velp::k4': [k_b, LogNorm(1e+1, 1e+12), get_cmap('CMRmap_r'),
        r'$P_{vel}/k_{B}$', 'K$\,$cm$^{-3}$', 'Turbulent Pressure', 'vol'],
    'gravp::k4': [-1.*k_b, LogNorm(1e+1, 1e+12), get_cmap('CMRmap_r'),
        r'$-P_{G}/k_{B}$', 'K$\,$cm$^{-3}$', 'Gravitational Pressure', 'vol'],
    'p_sum': [1., Normalize(-2e-10, 1e-10), get_cmap('coolwarm_r'),
        r'$P_{sum}$', 'barye', 'Total Pressure', 'vol'],
    'crossp': [1., LogNorm(1e-16, 1e-10), get_cmap('CMRmap_r'),
        r'$P_{ram}$', 'barye', 'Crossflow Ram Pressure', 'vol'],
    't_ff': [Myr, LogNorm(.3, 300.), get_cmap('CMRmap'),
        r'$t_{ff}$', 'Myr', 'Free Fall Time', 'mass'],    
    'l_jeans': [pc, LogNorm(.1, 1e+4), get_cmap('CMRmap'),
        r'$l_{jeans}$', 'pc', 'Jeans Length', 'mass'],
    'jeans_res': [1., LogNorm(1e-2, 1e+6), get_cmap('jet_r'),
        r'$l_{jeans}$/$l_{cell}$', 'ratio', 'Jeans Resolution Number', 'mass'],
    'velx': [km, Normalize(-20., 20.), get_cmap('Spectral'),
        r'$v_{x}$', 'km/s', 'X-Velocity', 'mass'],
    'velx::low': [km, Normalize(-5., 5.), get_cmap('RdBu'),
        r'$v_{x}$', 'km/s', 'X-Velocity', 'mass'],
    'vely': [km, Normalize(-20., 20.), get_cmap('Spectral'),
        r'$v_{y}$', 'km/s', 'Y-Velocity', 'mass'],
    'velz': [km, Normalize(-20., 20.), get_cmap('Spectral'),
        r'$v_{z}$', 'km/s', 'Z-Velocity', 'mass'],
    'vel': [km, Normalize(0., 17.), get_cmap('viridis_r'),
        r'$\left|v\right|$', 'km/s', 'Total Velocity', 'mass'],
    'vel::low': [km, Normalize(0., 5.), get_cmap('Spectral'),
        r'$\left|v\right|$', 'km/s', 'Total Velocity', 'mass'],
    'vel_yz': [km, Normalize(0., 8.), get_cmap('Spectral'),
        r'$\left|v_{yz}\right|$', 'km/s', 'Total Velocity', 'mass'],

    'vel_sq': [km**2, PowerNorm(.5, 0., 225.), get_cmap('Spectral'),
        r'$\left|v^2\right|$', 'km$^{2}$/s$^{2}$', 'Squared Velocity', 'mass'],
    'vel_yz_sq': [km**2, PowerNorm(.5, 0., 144.), get_cmap('Spectral'),
        r'$\left|v_{yz}^2\right|$', 'km$^{2}$/s$^{2}$', 'Squared Transversal Velocity', 'mass'],

    'mag_sq': [1e-12, PowerNorm(.5, 0., 100.), get_cmap('Spectral'),
        r'$B^{2}$', '$\mu$G$^2$', 'Squared Magnetization', 'vol'],

    'magx': [1e-6, Normalize(-15., 15.), get_cmap('Spectral'),
        r'$B_{x}$', '$\mu$G', 'X-Magnetization', 'vol'],
    'magx::pos': [1e-6, Normalize(0., 10.), get_cmap('Spectral'),
        r'$B_{x}$', '$\mu$G', 'X-Magnetization', 'vol'],
    'magy': [1e-6, Normalize(-15., 15.), get_cmap('Spectral'),
        r'$B_{y}$', '$\mu$G', 'Y-Magnetization', 'vol'],
    'magz': [1e-6, Normalize(-15., 15.), get_cmap('Spectral'),
        r'$B_{z}$', '$\mu$G', 'Z-Magnetization', 'vol'],
    'mag': [1e-6, LogNorm(1e-3, 1e+3), get_cmap('viridis_r'),
        r'$\left|B\right|$', '$\mu$G', 'Magnetization', 'vol'],
    'mag_yz': [1e-6, Normalize(0., 20.), get_cmap('Spectral'),
        r'$\left|B_{yz}\right|$', '$\mu$G', 'Magnetization', 'vol'],
    'beta': [1., LogNorm(1e-2, 1e+4), get_cmap('CMRmap_r'),
        r'$\beta$', 'ratio', 'Plasma Beta', 'vol'],
    'beta::m': [1., LogNorm(1e-1, 1e+1), get_cmap('RdBu'),
        r'$\beta$', 'ratio', 'Plasma Beta', 'mass'],
    'beta_v': [1., LogNorm(1e-1, 1e+1), get_cmap('CMRmap_r'),
        r'$\beta$', 'ratio', 'Velocity Beta', 'vol'],
    'rlevel': [1., Normalize(3.5, 8.5), get_cmap('jet'),
        r'$R$', 'ratio', 'Refinement Level', 'vol'],
    'epot': [-1., LogNorm(1e+10, 1e+12), get_cmap('skinhelix_r'),
        r'$-\Phi_{G}$ (shifted)', 'erg/g', 'Gravitational Potential', 'mass'],
    'gpot': [1., LogNorm(1e+10, 1e+12), get_cmap('skinhelix_r'),
        r'$\Phi_{G}$', 'erg/g', 'Gravitational Potential', 'mass'],
    'gpot::lin': [1., Normalize(-1e+11, 1e+11), get_cmap('RdYlBu'),
        r'$\Phi_{G}$', 'erg/g', 'Gravitational Potential', 'mass'],
    'gpot::lin2': [1., Normalize(-2e+11, 4e+10), get_cmap('skinhelix'),
        r'$\Phi_{G}$', 'erg/g', 'Gravitational Potential', 'mass'],
    'gpot::linS': [1., Normalize(-1e+10, 1e+10), get_cmap('RdYlBu'),
        r'$\Phi_{G}$', 'erg/g', 'Gravitational Potential', 'mass'],
    'vb_oangle': [1., Normalize(0., 180.), get_cmap('jet'),
        r'$\angle\left(v,B\right)$', 'degree', 'Oriented Velocity-Magnetization-Angle', 'vol'],
    'vb_angle': [1., Normalize(0., 90.), get_cmap('jet'),
        r'$\angle\left(v,B\right)$', 'degree', 'Velocity-Magnetization-Angle', 'vol'],
    'mach_s': [1., LogNorm(1e-1, 1e+2), get_cmap('CMRmap_r'),
        r'$v$/$c_{s}$', 'ratio', 'Mach Number', 'vol'],
    'mach_a': [1., LogNorm(1e-1, 1e+2), get_cmap('CMRmap_r'),
        r'$v$/$c_{a}$', 'ratio', 'Alfvén Number', 'vol'],
    'cdto': [1., Normalize(0., 3.), get_cmap('CMRmap_r'),
        r'$A_{v}$', 'mag', 'Visual Extinction', 'mass'],
    'cdto::high': [1., Normalize(0., 6.25), get_cmap('CMRmap_r'),
        r'$A_{v}$', 'mag', 'Visual Extinction', 'mass'],
    'cdto::low': [1., Normalize(0., 1.), get_cmap('CMRmap_r'),
        r'$A_{v}$', 'mag', 'Visual Extinction', 'mass'],
    'cdto::uh': [1., Normalize(0., 20.), get_cmap('CMRmap_r'),
        r'$A_{v}$', 'mag', 'Visual Extinction', 'mass'],
    'ionrate': [1., LogNorm(1.5e-4, 1.5e-2), get_cmap('CMRmap_r'),
        r'$2n_{e}$/$n$', 'ratio', 'Ionization Rate', 'parts'],
    'eint': [1., LogNorm(3e+7, 3e+12), get_cmap('CMRmap_r'),
        r'$e$', 'erg/g', 'Internal Energy', 'vol'],
    'vol': [pc**3, LogNorm(.8*(1./64.)**3, 5.*32.**3), get_cmap('CMRmap_r'),
        r'$V$', 'pc$^{3}$', 'Volume', None],
    'parts': [1., LogNorm(1e-10*pc**3, 1e+3*pc**3), get_cmap('CMRmap_r'),
        r'$N$', 'counts', 'Parts', None],
    'mass': [M_sol, LogNorm(1e-10, 8e+4), get_cmap('CMRmap_r'),
        r'$M$', 'M$_\odot$', 'Mass', None],
    'mass_h1': [M_sol, LogNorm(1e-10, 1.5e+5), get_cmap('CMRmap_r'),
        r'$M_{H1}$', 'M$_\odot$', 'H$_{a}$,H$_{+}$ Mass', None],
    'mass_h2': [M_sol, LogNorm(1e-10, 4.0e+4), get_cmap('CMRmap_r'),
        r'$M_{H2}$', 'M$_\odot$', 'H$_{2}$ Mass', None],
    'mass_cp': [M_sol, LogNorm(1e-10, 1.5e+4), get_cmap('CMRmap_r'),
        r'$M_{C+}$', 'M$_\odot$', 'C$_{+}$ Mass', None],
    'mass_co': [M_sol, LogNorm(1e-12, 15.), get_cmap('CMRmap_r'),
        r'$M_{CO}$', 'M$_\odot$', 'CO Mass', None],

    'massrate_h2': [M_sol/Myr, Normalize(-.05, .05), get_cmap('CMRmap_r'),
        r'$\dot M_{H2}$', 'M$_\odot$/Myr', 'H$_{2}$ Formation Rate', None],
    'massrate_h2::sym': [M_sol/Myr, SymLogNorm(3., 1., -5.e+2, 3.e+3), get_cmap('CMRmap_r'),
        r'$\dot M_{H2}$', 'M$_\odot$/Myr', 'H$_{2}$ Formation Rate', None],
    'massrate_h2::pos': [M_sol/Myr, LogNorm(3e-6, 3e+3), get_cmap('CMRmap_r'),
        r'$\dot M_{H2}$', 'M$_\odot$/Myr', 'H$_{2}$ Formation Rate', None],
    'massrate_h2::neg': [-M_sol/Myr, LogNorm(1e-7, 1e+2), get_cmap('CMRmap_r'),
        r'$\dot M_{H2}$', 'M$_\odot$/Myr', 'H$_{2}$ Destruction Rate', None],

    'massrate_co': [M_sol/Myr, SymLogNorm(5e-3, 1., -.1, .1), get_cmap('RdGy'),
        r'$\dot M_{CO}$', 'M$_\odot$/Myr', 'CO Formation Rate', None],
    'massrate_co::pos': [M_sol/Myr, LogNorm(3e-8, 3e+1), get_cmap('CMRmap_r'),
        r'$\dot M_{CO}$', 'M$_\odot$/Myr', 'CO Formation Rate', None],
    'massrate_co::neg': [-M_sol/Myr, LogNorm(5.5e-8, 5.5e+1), get_cmap('CMRmap_r'),
        r'$\dot M_{CO}$', 'M$_\odot$/Myr', 'CO Destruction Rate', None],

    'massfrac_hp': [1., Normalize(0., 1.), get_cmap('CMRmap_r'),
        r'$f_{H+}$', 'fraction', 'H$_{+}$ Mass Fraction', None],
    'massfrac_ha': [1., Normalize(0., 1.), get_cmap('CMRmap_r'),
        r'$f_{H_a}$', 'fraction', 'H$_{a}$ Mass Fraction', None],
    'massfrac_h2': [1., Normalize(0., 1.), get_cmap('CMRmap_r'),
        r'$f_{H_2}$', 'fraction', 'H$_{2}$ Mass Fraction', None],
    'massfrac_cp': [1., Normalize(0., 1e-3), get_cmap('CMRmap_r'),
        r'$f_{C+}$', 'fraction', 'H$_{+}$ Mass Fraction', None],
    'massfrac_co': [1., Normalize(0., 1e-3), get_cmap('CMRmap_r'),
        r'$f_{CO}$', 'fraction', 'H$_{CO}$ Mass Fraction', None],
    'massfrac_h1': [1., Normalize(0., 1.), get_cmap('CMRmap_r'),
        r'$f_{H_1}$', 'fraction', 'H$_{1}$ Mass Fraction', None],

    'f_hp': [1., Normalize(0., 1.), get_cmap('CMRmap_r'),
        r'$f_${H+}$=\frac{}{n_{Hx}}$', 'fraction', 'H$_{+}$ Mass Fraction', None],
    'f_ha': [1., Normalize(0., 1.), get_cmap('CMRmap_r'),
        r'$f_${H_a}$=\frac{}{n_{Hx}}$', 'fraction', 'H$_{a}$ Mass Fraction', None],
    'f_h2': [1., Normalize(0., 1.), get_cmap('CMRmap_r'),
        r'$f_${H_2}$=\frac{}{n_{Hx}}$', 'fraction', 'H$_{2}$ Mass Fraction', None],
    'f_cp': [1., Normalize(0., 1.), get_cmap('CMRmap_r'),
        r'$f_${C+}$=\frac{n_{C+}}{n_{Cx}}$', 'fraction', 'H$_{+}$ Mass Fraction', None],
    'f_co': [1., Normalize(0., 1.), get_cmap('CMRmap_r'),
        r'$f_${CO}$=\frac{n_{CO}}{n_{Cx}}$', 'fraction', 'H$_{CO}$ Mass Fraction', None],

        
    'vorx': [1., Normalize(-1e-12, 1e-12), get_cmap('Spectral'),
        r'$\omega_{x}$', '1/s', 'X-Vorticity', 'mass'],
    'vory': [1., Normalize(-1e-12, 1e-12), get_cmap('Spectral'),
        r'$\omega_{y}$', '1/s', 'Y-Vorticity', 'mass'],
    'vorz': [1., Normalize(-1e-12, 1e-12), get_cmap('Spectral'),
        r'$\omega_{z}$', '1/s', 'Z-Vorticity', 'mass'],
    'vort': [1., LogNorm(1e-16, 1e-11), get_cmap('CMRmap_r'),
        r'$\left|\omega\right|$', '1/s', 'Total Vorticity', 'mass'],
    'vort_x': [1., LogNorm(1e-16, 1e-11), get_cmap('CMRmap_r'),
        r'$\left|\omega_{x}\right|$', '1/s', 'X-Vorticity', 'mass'],
    'vort_y': [1., LogNorm(1e-16, 1e-11), get_cmap('CMRmap_r'),
        r'$\left|\omega_{y}\right|$', '1/s', 'Y-Vorticity', 'mass'],
    'vort_z': [1., LogNorm(1e-16, 1e-11), get_cmap('CMRmap_r'),
        r'$\left|\omega_{z}\right|$', '1/s', 'Z-Vorticity', 'mass'],
    'vort_yz': [1., LogNorm(1e-16, 1e-11), get_cmap('CMRmap_r'),
        r'$\left|\omega_{yz}\right|$', '1/s', 'YZ-Vorticity', 'mass'],

    'divb': [1e-6/pc, Normalize(-1.5, 1.5), get_cmap('RdBu'),
        r'$\nabla B$', '$\mu$G/pc', 'Magnetic Field Divergence', 'vol'],
    'div_b': [1e-6/pc, LogNorm(1e-4, 1e+1), get_cmap('CMRmap_r'),
        r'$\left|\nabla B\right|$', '$\mu$G/pc', 'Magnetic Field Divergence', 'vol'],

    'DEFAULT': [1., LogNorm(1e-5, 1e+5), get_cmap('CMRmap_r'),
        r'VARSIGN', 'code units', 'VARNAME', 'vol'],        
}

var_keys = list(var_settings.keys())
for var in var_keys:
    assert len(var_settings[var]) == 7
    if 'dx_' in var: continue
    if 'dy_' in var: continue
    if 'dz_' in var: continue
    unit, norm, cmap, label, ulabel, title, cweight = var_settings[var]
    unit_d = unit/pc
    label_dx = r'$\frac{d}{dx}$ ' +label
    label_dy = r'$\frac{d}{dy}$ ' +label
    label_dz = r'$\frac{d}{dz}$ ' +label
    ulabel_d = ulabel +'/pc'
    title_dx = title +' Grad-X'
    title_dy = title +' Grad-Y'
    title_dz = title +' Grad-Z'
    if not 'dx_'+var in var_keys:
        var_settings['dx_'+var] = [unit_d, norm, cmap, label_dx, ulabel_d, title_dx, cweight]
    if not 'dy_'+var in var_keys:
        var_settings['dy_'+var] = [unit_d, norm, cmap, label_dy, ulabel_d, title_dy, cweight]
    if not 'dz_'+var in var_keys:
        var_settings['dz_'+var] = [unit_d, norm, cmap, label_dz, ulabel_d, title_dz, cweight]

def GetVarSettings(key_in):
    varkey = key_in.split('::')[0]
    if key_in in var_settings:
        vs = var_settings[key_in]
    elif varkey in var_settings:
        vs = var_settings[varkey]
    else:
        print(f'WARNING: No settings for variable {varkey} specified. Using defaults.')
        vs = var_settings['DEFAULT']
        vs[3] = varkey
        vs[5] = 'var %s'%varkey
    return vs
