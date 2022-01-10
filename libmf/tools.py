# -*- coding: utf-8 -*-
import fnmatch,os,sys
import pickle
import gzip
import time
import numpy as np
import matplotlib.pylab as plt
from collections import OrderedDict
from ast import literal_eval
import csv
import pickle

__author__ = "Michael Weis"
__version__ = "0.9.5.0"

# ==== CONSTANTS ===============================================================

datadir = 'cldata'

homepath = os.path.expanduser("~")
wall = str(int(time.time()))


#===============================================================================
# ==== HELPER ==================================================================
#===============================================================================

def select(cluster):
    return tuple(map(np.array, list(zip(*cluster))))

def modclip(a, a_min, a_max):
    return a_min + np.mod(a-a_min, a_max-a_min)

def tstring(hdfdata):
    time = hdfdata['real scalars/time'] / Myr
    return r'$t=$'+'%.2f'%time+r'$\,$Myr'

def compose_name(*scraps):
    figname = ''
    for part in scraps:
        if part is not None:
            if figname:
                figname += '-'
            figname += str(part)
    return figname
    
#===============================================================================
# ==== GENERAL FIGURE TOOLS ====================================================
#===============================================================================

# ==== SETTINGS ================================================================
pickle_fileext = '.pkl'
fig_basepath = os.path.join(homepath, 'plots')
fig_outdir = wall
fig_target = 'display'

# ==== PRESETS =================================================================

mnras_ticks = {
    'direction': 'in',
    'labelsize': 12,
    'bottom': True,
    'left': True,
    'top': True,
    'right': True,
    'labelbottom': True,
    'labelleft': True,
    'labeltop': False,
    'labelright': False,
}

mnras_ticks_small = {}
mnras_ticks_small.update(mnras_ticks)
mnras_ticks_small.update({'labelsize': 10,})

mnras_matrix_ticks = {
    'direction': 'in',
    'labelsize': 12,
    'bottom': True,
    'left': True,
    'top': True,
    'right': True,
}

fig_presets = {
    'enum': {
        'figsize': (38.40, 21.60),
        'figdpi': 150,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': True,
        'extension': '.png',
        'bbox_inches': None,
        'enum_marker': True,
        }, 
    'display': {
        'figsize': (25.60, 14.40),
        'figdpi': 150,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': None,
        'enum_marker': False,
        }, 
    'thesis': {
        'figsize': (8.00, 6.00),
        'figdpi': 300,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': False,
        'counter_enable': True,
        'extension': '.pdf',
        'bbox_inches': None,
        'enum_marker': False,
        }, 
    'presentation': {
        'figsize': (12.00, 9.00),
        'figdpi': 300,
        'fontsize': 14,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': False,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis': {
        'figsize': (5.70, 4.30),
        'figdpi': 300,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis_half': {
        'figsize': (3.0, 4.30),
        'figdpi': 300,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis_fine': {
        'figsize': (8.55, 6.45),
        'figdpi': 300,
        'fontsize': 14,
        'legend_fontsize': 14,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis_half_fine': {
        'figsize': (4.5, 5.2),
        'figdpi': 300,
        'fontsize': 14,
        'legend_fontsize': 14,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis_hfl': {
        'figsize': (4.5, 5.2),
        'figdpi': 300,
        'fontsize': 14,
        'legend_fontsize': 14,
        'legend_enable': True,#False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis_large_fine': {
        'figsize': (8.55, 9.00),
        'figdpi': 300,
        'fontsize': 14,
        'legend_fontsize': 14,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis_large_fine2': {
        'figsize': (8.55, 9.00),
        'figdpi': 300,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mnras_page': {
        'figsize': (14.00, 10.50),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': None,
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_page_legend': {
        'figsize': (14.00, 10.50),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': None,
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_col': {
        'figsize': (6.60, 6.00),
        'figdpi': 600,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_col_legend': {
        'figsize': (6.60, 6.00),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 10,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_col_slegend': {
        'figsize': (6.60, 6.00),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 8,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_2col_legend': {
        'figsize': (13.20, 6.00),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 10,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_page': {
        'figsize': (2.*504./72.27, .72*2.*682./72.27),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_col_narrow': {
        'figsize': (6.60, 4.40),
        'figdpi': 700,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks_small,
        }, 
    'mnras_matrix': {
        'figsize': (.75*2.*504./72.27, .75*.72*2.*682./72.27),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_matrix_ticks,
        }, 

    'screen_matrix': {
        'figsize': (19.2, 10.8),
        'figdpi': 200,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_matrix_ticks,
        }, 
    'present_col': {
        'figsize': (8.00, 6.00),
        'figdpi': 600,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
}

def get_fig_preset(target=None):
    if target is None:
        target=fig_target
    return fig_presets[target]
    
def set_fig_preset(target='mnras_col_legend'):
    global fig_target
    fig_target = target
    fs = fig_presets[fig_target]
    plt.rcParams.update({'font.size': fs['fontsize']})
    return fs

# ==== AXIS TOOLS ==============================================================
def ax_title(ax, title):
    # Load figure settings
    fs = fig_presets[fig_target]
    if isinstance(ax, list):
        #for axi in ax:
        #    plot_title(axi, title)
        return ax_title(ax[0], title)
    # Set title if figure titles are enabled:
    elif fs['figtitle_enable']:
        ax.set_title(title)
        return True
    else:
        return False

def ax_format_ticks(ax, axis='both'):
    # Load figure settings
    fs = fig_presets[fig_target]
    #
    kwargs = fs.get('tick_params', dict())
    ax.tick_params(axis=axis, which='both', **kwargs)
    return kwargs
    
def ax_legend(ax, *args, **kwargs):
    fs = fig_presets[fig_target]
    if not fs['legend_enable']:
        return None
    prop = dict(size=fs['legend_fontsize'])
    if not prop in kwargs:
        kwargs['prop'] = prop
    else:
        kwargs['prop'].update(prop)
    lg = ax.legend(*args, **kwargs)
    return lg    

def axFakeLabel(ax, text, *args, **kwargs):
    ''' Add a specific label to an ax legend without plotting anything '''
    kwargs.update(dict(label=text))
    ax.plot([np.nan, np.nan], [np.nan, np.nan], *args, **kwargs)

# ==== SAVE FIGURE =============================================================
def slugify(value):
    import re
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    #value = normalize('NFKD', value).encode('ascii', 'ignore')
    value = re.sub('[^\w\s-]', '', value).strip()
    value = re.sub('[-\s]+', '-', value)
    return value
    
def get_savefig_target(name, subpath=''):
    fs = fig_presets[fig_target]
    figpath = os.path.join(fig_basepath, fig_outdir)
    # Get directory to save file to
    if fs['subpath_enable']:
        filedir = os.path.join(figpath, subpath)
    else:
        filedir = figpath
    # Add counter prefix to filename
    if fs['counter_enable']:
        if not subpath in savefig.counter:
            savefig.counter[subpath] = 0
        savefig.counter[subpath] += 1
        cntstr = '%i_-_'%savefig.counter[subpath]
    else:
        cntstr = ''
    # Assemble filename
    if fs['subpath_enable']:
        filename = cntstr +name
    else:
        subpathname = slugify(subpath)
        filename = subpathname +'_-_' +cntstr +name           
    #
    return filedir, filename, fs['extension']
    
def get_savefig_filepath(name, subpath=''):
    # Get unused filepath in guaranteed directory
    filedir, filename, fileext = get_savefig_target(name, subpath=subpath)
    # Assemble path to file
    filepath = os.path.join(filedir, filename+fileext)
    return filepath

def sanitize_save_target(filedir, filename, fileext):
    # Assemble path to file
    filepath = os.path.join(filedir, filename+fileext)
    # Rename if a file with the same name already exists:
    i = 1
    while os.path.exists(filepath):
        i += 1
        filename_new = filename +'_V%i'%i
        filepath = os.path.join(filedir, filename_new+fileext)
    # Make target directory if non-existent
    makedir(os.path.dirname(filepath))
    return filepath
    
def find_newest_target(filedir, filename, fileext):
    filepath_exist = None
    # Assemble path to file
    filepath = os.path.join(filedir, filename+fileext)
    # Rename while a file with the same name already exists:
    i = 1
    while os.path.exists(filepath):
        filepath_exist = filepath
        i += 1
        filename_new = filename +'_V%i'%i
        filepath = os.path.join(filedir, filename_new+fileext)
    # 
    return filepath_exist
    
def saveimg(img, name, subpath='', raw_gamma=2.2, **overrides):
    assert isinstance(raw_gamma, float)
    from imageio import imwrite
    # Get unused filepath in guaranteed directory
    filedir, filename, fileext = get_savefig_target(name, subpath=subpath)
    fileext = '.png'
    filepath = sanitize_save_target(filedir, filename, fileext)
    # Write image raw data
    data = img.get_array()
    try:
        cmap = img.get_cmap()
        norm = img.norm
        imwrite(filepath, cmap(norm(data)), **overrides)
    except:
        print(f'#D#raw_gamma: {raw_gamma}')
        data_raw = data**(1./raw_gamma)
        imwrite(filepath, data_raw, **overrides)
    return filepath
    

def savefig(fig, name, subpath='', autoclose=True, autopickle=False, **overrides):
    # Load figure settings
    fs = fig_presets[fig_target]
    # Get unused filepath in guaranteed directory
    filedir, filename, fileext = get_savefig_target(name, subpath=subpath)
    filepath = sanitize_save_target(filedir, filename, fileext)
    # Set figure parameters
    # figsize is ignored by savefig, but set as a parameter anyways,
    # because the user can then use the override to control the figure size.
    savefig_param = dict(dpi=fs['figdpi'], bbox_inches=fs['bbox_inches'],
        figsize=fs['figsize'])
    savefig_param.update(overrides)
    # Pop the figure size from the dictionary, which is always present,
    # as per above construction.
    # This is necessary because savefig now throws an error if given a figsize parameter
    fig_width, fig_height = savefig_param.pop('figsize')
    # Set figure size, like specified in the parameter dict.
    fig.set_size_inches(fig_width, fig_height, True)
    #print fig_width, fig_height
    #print savefig_param
    # Write figure
    print(f'-> Exporting {filepath}')
    fig.savefig(filepath, **savefig_param)
    # Store figure as pickle for later usage (e.g. combining figures)
    if autopickle:
        storefig(fig, name, subpath=subpath, autoclose=False)
    # Autoclose figure
    if autoclose:
        plt.close(fig)
    return filepath

def storefig(fig, name, subpath='', autoclose=False):
    # Get unused filepath in guaranteed directory
    filedir, filename, fig_fileext = get_savefig_target(name, subpath=subpath)
    filepath = sanitize_save_target(filedir, filename, pickle_fileext)
    # Pickle figure
    print(f'-> Storing {filepath}')
    with file(filepath, 'w') as f:
        pickle.dump(fig, f)
    # Autoclose figure
    if autoclose:
        plt.close(fig)
    #
    return filepath
    
def restorefig(name, subpath=''):
    filedir, filename, fig_fileext = get_savefig_target(name, subpath=subpath)
    filepath = find_newest_target(filedir, filename, pickle_fileext)
    filepath_expected = os.path.join(filedir, filename+pickle_fileext)
    if filepath is None:
        raise ValueError('Not Found: Pickled Figure %s'%filepath_expected)
    else:
        fig = pickle.load(filepath)
        #TODO: ADD CHECK: IS RETRIEVED OBJECT ACTUALLY A FIGURE?
    return fig

savefig.counter = dict()


#===============================================================================
# ==== PARAMETER FILE PARSER ===================================================
#===============================================================================

def safe_eval(value):
    value_dict = {'.true.': True, '.false.': False}
    if value in value_dict:
        return value_dict[value]
    try:
        # known to crash on '.true.', '.false.', may go wrong in other cases.
        return literal_eval(value)
    except:
        # no clue how to evaluate (e.g. string like 'periodic'). Leave it as is.
        return value
        
def readpar(filename):
    par_dict = {'plot_vars': set()}
    with open(filename, 'r') as infile:
        for line in infile:
            # cut line prior to first # :
            line = line.rsplit('#')[0]
            # omit line if not containing = :
            if '=' not in line:
                continue
            # Transfer key and value from line to dictionary:
            key, foo, value = [word.strip() for word in line.rpartition('=')]
            if key.startswith('plot_var_'):
                par_dict['plot_vars'].add(value)
            else:
                par_dict[key] = safe_eval(value)
    return par_dict
    
    
#===============================================================================
# ==== FINGERPRINTING ==========================================================
#===============================================================================

def get_plotfile_fingerprint(hdfdata):
    hashable_repr = (hdfdata['dens']).tostring()
    hdf_fp = format(hash(hashable_repr), 'x')
    return hdf_fp
    
def get_array_fingerprint(a):
    hashable_repr = a.tostring()
    fingerprint = format(hash(hashable_repr), 'x')
    return fingerprint

#===============================================================================
# ==== HUMAN READABLE DATA I/O =================================================
#===============================================================================

def save_csv(dictlist, outfile, force=False):
    ''' Write a list of dictionaries containing value sets into a human-readable text file '''
    # Make sure target directory exists
    filedir = os.path.dirname(outfile)
    filedir = makedir(os.path.dirname(outfile))
    # Avoid overwriting an already existing file
    if os.path.exists(outfile) and not force:
        raise IOError('Output file already exists: %s' % outfile)
    # Write csv file
    keys = list(dictlist[0].keys())
    with open(outfile, 'wb') as f:
        print(f'File (W-CSV): {outfile}')
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(dictlist)
    #
    return outfile
    
def load_csv(infile):
    dictlist = list()
    with open(infile, mode='rb') as f:
        print(f'File (R-CSV): {infile}')
        reader = csv.DictReader(f)
        for row in reader:
            dictlist.append(row)
    return dictlist
    
def load_csv_cols(infile):
    dictlist = list()
    with open(infile, mode='rb') as f:
        print(f'File (R-CSV-cols): {infile}')
        reader = csv.DictReader(f)
        varkeys = reader.fieldnames
        for row in reader:
            dictlist.append(row)
    vardict = OrderedDict()
    for var in varkeys:
        vardict[var] = np.array([safe_eval(row[var]) for row in dictlist])
    return vardict


#===============================================================================
# ==== GENERALIZED FILE I/O ====================================================
#===============================================================================

def makedir(filedir):
    # Try to create datafile target directory; 
    # if this fails, the directory should already be there.
    # Otherwise something is seriously wrong!
    # (File with dirname? No write access?)
    try: 
        os.makedirs(filedir)
    except OSError:
        if not os.path.isdir(filedir):
            raise
    return filedir

def save_data(data, outfile, force=False, compress=False):
    filedir = os.path.dirname(outfile)
    filedir = makedir(os.path.dirname(outfile))
    # Create a preliminary filename to write the file (rename after write);
    # because the write-process may be incomplete due to error/interruption,
    # to avoid yielding a defective file of the desired name then.
    prefile = outfile +'.pre'
    # Avoid overwriting an already existing file
    if os.path.exists(outfile) and not force:
        raise IOError('Output file already exists: %s' % outfile)
    # Write data to preliminary file
    if compress:
        with gzip.open(prefile, 'wb') as f:
            print(f'File (WC): {outfile}')
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(prefile, 'wb') as f:
            print(f'File (W): {outfile}')
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    # Rename preliminary file to desired filename
    os.rename(prefile, outfile)
    return outfile

def load_data(infile):
    if not os.path.exists(infile):
        raise IOError('Input file does not exist: %s' % infile)
    try:
        with open(infile, 'rb') as f:
            print(f'File (R): {infile}')
            data = pickle.load(f)
    except:
        with gzip.open(infile, 'rb') as f:
            print(f'File (RC): {infile}')
            data = pickle.load(f)
    return data

def get_datafile_path(hdfdata, prefix, suffix='', extension='.pkl', dirname=None):
    # Build path to data file from plotfile path
    plotfile = hdfdata.filename()
    basename = os.path.basename(plotfile).replace('_hdf5', '')
    filename = prefix +basename +suffix +extension
    if dirname is None:
        basedir = os.path.dirname(plotfile)
        dirname = os.path.join(basedir, datadir)
    filepath = os.path.join(dirname, filename)
    return filepath

    
#===============================================================================
# ==== DIRECTORY TRAVERSAL =====================================================
#===============================================================================

def scanpath(simpath, pattern, exclude_pattern='*_forced_*'):
    dirdict = dict()
    if os.path.isfile(simpath):
        dirpath = os.path.dirname(simpath)
        filename = os.path.basename(simpath)
        dirdict[dirpath] = [filename,]
    elif os.path.isdir(simpath):
        for dirpath, dirnames, files in os.walk(simpath, followlinks=True):
            if datadir in dirpath:
                continue
            filenames_include = set(fnmatch.filter(files, pattern))
            filenames_exclude = set(fnmatch.filter(files, exclude_pattern))
            filenames = list(filenames_include-filenames_exclude)
            if len(filenames) == 0:
                continue
            filenames.sort()
            dirdict[dirpath] = filenames
    return dirdict

