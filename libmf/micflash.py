# -*- coding: utf-8 -*-
import numpy as np
from .micvar import plotfile, flashfile, var_grid, var_mhd, var_ch5, var_ch15, var_2d
import sys
from math import floor, ceil
from copy import deepcopy
import libmf.blocktree as blocktree
#from tricubic_hermite import tricubic
from .tricubic_direct import tricubic


__author__ = "Michael Weis"
__version__ = "1.0.0.1"

# ==== HELPER ==================================================================
def rangeselect(L, L_min, L_max):
    a = np.array(L)
    mask = (a>=L_min)*(a<L_max)
    return a[mask]

def modclip(a, a_min, a_max):
    return a_min + np.mod(a-a_min, a_max-a_min)
    
def pscale(X):
    """
    Rank data from 0.0 to 1.0 using cubic rank spacing.
    Useful for displaying data of unknown/irregular scaling behaviour.
    """
    S = np.sort(X, axis=None)
    R = np.searchsorted(S, X)
    P = R*1./R.max()
    return P**3

# ==== PARAMETER READER ========================================================
def ff_lmax(ffile, default=None):
    try:
        ljeans_max = ffile['integer runtime parameters/lrefine_max_jeans']
        lstd_max = ffile['integer runtime parameters/lrefine_max']
        refine_max = max(ljeans_max, lstd_max)
    except KeyError:
        try:
            refine_max = ffile['integer runtime parameters/lrefine_max']
        except KeyError:
            refine_max = default
    return refine_max
    
def ff_usegrav(ffile, default=True):
    try:
        sim_gravity = ffile['logical runtime parameters/usegravity']
    except KeyError:
        # Workaround for h5py bug not reading 'logical runtime parameters' in 2.60:
        try:
            gpot = ffile['gpot']
            sim_gravity = ( gpot.min() != gpot.max() )
        except KeyError:
            sim_gravity = default
    return sim_gravity


################################################################################
##### CLASS: flat leaf-level blocktree (mainly 1D / 2D / uniform 3D)  ##########
################################################################################

class FlatTree(object):
    def __init__(self, block_bbox, nodetype):
        # 3D Array block_bbox[nblock,dim.(x/y/z),orient.(min/max)]
        # 1D Array nodetype[nblock] (=1 for leafs, =0 for inner blocks)
        
        leafblock_mask = (nodetype == 1)
        lfnblock = np.arange(len(nodetype))[leafblock_mask]

        self._bbox_min = block_bbox[:,:,0]
        self._bbox_max = block_bbox[:,:,1]
        self._bbox_size = self._bbox_max -self._bbox_min

        self._leaf_size = self._bbox_size.min(axis=0)
        self._domain_min = self._bbox_min.min(axis=0)
        self._domain_max = self._bbox_max.max(axis=0)
        self._domain_size = self._domain_max - self._domain_min
        #print self._domain_min, self._domain_max
        
        #pivot_dim = np.argmax(self._leaf_size)
        #trivial_dims = self._leaf_size < 1e-8*self._leaf_size[pivot_dim]
        #self._leaf_size[trivial_dims] = self._leaf_size[pivot_dim]
        #self._domain_size[trivial_dims] = self._leaf_size[pivot_dim]
        #self._domain_min[trivial_dims] -= .5*self._leaf_size[pivot_dim]
        #print 'FlatTree: Trivial Dims:', trivial_dims
        
        domainres_f = self._domain_size / self._leaf_size
        #print domainres_f
        self._domainres = list((domainres_f+.5).astype(np.int32))
        
        bbox_fimin = (self._bbox_min -self._domain_min) / self._leaf_size
        bbox_fimax = (self._bbox_max -self._domain_min) / self._leaf_size

        lfbbox_imin = (bbox_fimin[leafblock_mask] +.5).astype(np.int32)
        lfbbox_iend = (bbox_fimax[leafblock_mask] +.5).astype(np.int32)
            
        self._leafblock_LUT = np.full(self._domainres, -1)
        for nbl, imin, iend in zip(lfnblock, lfbbox_imin, lfbbox_iend):
            #print nbl, imin, iend
            self._leafblock_LUT[imin[0]:iend[0],imin[1]:iend[1],imin[2]:iend[2]] = nbl
            
        if np.any(self._leafblock_LUT < 0):
            raise RuntimeError('FlatTree Init: Incomplete Leaf LUT')

    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.Flush()
        return False
    def __del__(self):
        self.Flush()

    def Findblock(self, x, y, z):
        LUT_fix = ((x -self._domain_min[0]) / self._leaf_size[0])
        LUT_fiy = ((y -self._domain_min[1]) / self._leaf_size[1])
        LUT_fiz = ((z -self._domain_min[2]) / self._leaf_size[2])
        LUT_ix = np.clip(LUT_fix.astype(np.int32), 0, self._domainres[0]-1)
        LUT_iy = np.clip(LUT_fiy.astype(np.int32), 0, self._domainres[1]-1)
        LUT_iz = np.clip(LUT_fiz.astype(np.int32), 0, self._domainres[2]-1)
        nbl = self._leafblock_LUT[LUT_ix, LUT_iy, LUT_iz]
        return nbl
        
    def Flush(self):
        if self._leafblock_LUT is not None:
            self._leafblock_LUT = None
        return 1


################################################################################
##### CLASS: flash file 3d paramesh octree structure ###########################
################################################################################

class pm3dgrid(object):
    def __init__(self, flashfile, verbose=True, force_flat_tree=False):
        if verbose:
            print('Reading Layout Data...')
            sys.stdout.flush()
        block_bbox = flashfile['bounding box']
        self.__nxb = flashfile['integer scalars/nxb']
        self.__nyb = flashfile['integer scalars/nyb']
        self.__nzb = flashfile['integer scalars/nzb']
        self.__rlevel = flashfile['refine level']
        self.__ntype = flashfile['node type']
        if verbose:
            print('Processing Layout Data...')
            sys.stdout.flush()
        self._nb = np.array((self.__nxb, self.__nyb, self.__nzb))
        self._blockcells = self.__nxb*self.__nyb*self.__nzb
        self._bbox_min = block_bbox.min(axis=2)
        self.__block_xmin = self._bbox_min[:,0]
        self.__block_ymin = self._bbox_min[:,1]
        self.__block_zmin = self._bbox_min[:,2]
        self._bbox_max = block_bbox.max(axis=2)
        self._bbox_center = .5*(self._bbox_min+self._bbox_max)
        self._bbox_size = self._bbox_max - self._bbox_min
        self._bbox_minsize = self._bbox_size.min(axis=0)
        self._bbox_maxsize = self._bbox_size.max(axis=0)
        self._domain_min = self._bbox_min.min(axis=0)
        self._domain_max = self._bbox_max.max(axis=0)
        self._domain_size = self._domain_max - self._domain_min
        self._domain_blocks = (self._domain_size/self._bbox_minsize +.5).astype(np.int32)
        self._domain_cells = self._domain_blocks * self._nb
        # Precalculate (reciprocal) values to aleviate findblock/findblockcell
        self.__xmin = self._domain_min[0]
        self.__ymin = self._domain_min[1]
        self.__zmin = self._domain_min[2]
        self.__cell_xsize_recp = self.__nxb * 1./self._bbox_size[:,0]
        self.__cell_ysize_recp = self.__nyb * 1./self._bbox_size[:,1]
        self.__cell_zsize_recp = self.__nzb * 1./self._bbox_size[:,2]
        self.__block_xminsize_recp = 1./self._bbox_minsize[0]
        self.__block_yminsize_recp = 1./self._bbox_minsize[1]
        self.__block_zminsize_recp = 1./self._bbox_minsize[2]
        # Store information about dimensionality
        self.__trivial_dim = (self._nb == 1)
        self.__uniform_dim = (self._bbox_minsize / self._bbox_maxsize) > (1.-1e-4)
        ### Build a block index tree indicating the spacial block layout
        if np.any(self.__trivial_dim) or np.all(self.__uniform_dim) or force_flat_tree:
            if verbose:
                print('Building Block Index Field...')
                sys.stdout.flush()
            self._blocktree = FlatTree(block_bbox, self.__ntype.astype(np.uint64))
        else:
            if verbose:
                print('Building Block Index Octree...')
                sys.stdout.flush()
            self._blocktree = blocktree.blocktree(block_bbox, self.__ntype.astype(np.uint64))
        ###
        self.__cell_coords = None # postpone construction till needed
        self.__cellbox_min = None # postpone construction till needed
        self.__cellbox_max = None # postpone construction till needed
        self.__tree = None # postpone construction till needed
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self._blocktree.Flush()
        return False
    def __del__(self):
        try:
            self._blocktree.Flush()
        except AttributeError:
            pass
    def __build_cell_coords(self):
        if self.__cell_coords is None:
            ### Calculate coordinates of the cells inside the blocks
            x_reloffset = (np.linspace(-.5, .5, self.__nxb, endpoint=False) + .5/self.__nxb)
            y_reloffset = (np.linspace(-.5, .5, self.__nyb, endpoint=False) + .5/self.__nyb)
            z_reloffset = (np.linspace(-.5, .5, self.__nzb, endpoint=False) + .5/self.__nzb)
            reloffset = np.array(np.meshgrid(y_reloffset, z_reloffset, x_reloffset))[(2,0,1),:,:,:]
            offset = (self._bbox_size).reshape(-1,3,1,1,1)*reloffset
            self.__cell_coords = (self._bbox_center).reshape(-1,3,1,1,1) +offset
            cellbox_radius = (.5*self._bbox_size/self._nb).reshape(-1,3,1,1,1)
            self.__cellbox_min = self.__cell_coords - cellbox_radius
            self.__cellbox_max = self.__cell_coords + cellbox_radius
        else:
            pass            
    def __repr__(self, level=0):
        answer  = 'Flashfile 3D Paramesh AMR Grid' + '\n'
        answer += 'Block layout:  ' + str(self._domain_blocks) + ' '
        answer += '(' + str(len(self._bbox_min)) + ')' + '\n'
        answer += 'Block celling: ' + str(self._nb) + '\n'
        answer += 'Cell layout:   ' + str(self._domain_cells) + ' '
        answer += '(' + str(len(self._bbox_center)*self._blockcells) + ')' + '\n'
        return answer
        
    def blk_rlevel(self):
        return self.__rlevel
        
    def blk_ntype(self):
        return self.__ntype
        
    def blk_shape(self):
        return (len(self.__rlevel), self.__nzb, self.__nyb, self.__nxb)
        
    def cell_rlevel(self):
        shape_ones = np.ones(self.blk_shape(), dtype=np.int)
        return self.__rlevel[:,None,None,None] *shape_ones
    
        
    def coords(self, block=slice(None), axis=slice(None), ng=0):
        """ Get metrics of the bounding boxes of the blocks """
        # ng: number of guard cells on each side
        # Calculate relative block extent correction factors for guard cell consideration
        fx = (self.__nxb+2.*ng) / self.__nxb
        fy = (self.__nyb+2.*ng) / self.__nyb
        fz = (self.__nzb+2.*ng) / self.__nzb
        #
        x_reloffset = np.linspace(-.5*fx, .5*fx, self.__nxb+2*ng, endpoint=False) + .5/self.__nxb
        y_reloffset = np.linspace(-.5*fy, .5*fy, self.__nyb+2*ng, endpoint=False) + .5/self.__nyb
        z_reloffset = np.linspace(-.5*fz, .5*fz, self.__nzb+2*ng, endpoint=False) + .5/self.__nzb
        reloffset = np.array(np.meshgrid(y_reloffset, z_reloffset, x_reloffset))[(2,0,1),:,:,:]
        coords  = reloffset[axis] * self._bbox_size[block,axis,None,None,None]
        coords += self._bbox_center[block,axis,None,None,None]
        return coords
        
    def cellsize(self, block=slice(None), axis=None):
        if axis is None:
            return np.array((self.cellsize(block=block, axis=0),
                             self.cellsize(block=block, axis=1),
                             self.cellsize(block=block, axis=2)))
        elif not axis in (0,1,2):
            raise ValueError(f'Axis {axis} out of range(3).')
        return self._bbox_size[block,axis] / self._nb[axis]

    def fingerprint(self):
        hashable_repr = (self._bbox_center).tostring()
        return hash(hashable_repr)
        
# --- FIND BLOCK ID BY COORDINATE -------------------------------------------------
    def findblock(self, x, y, z):
        """
        Return the index of the block containing the coordinates x, y, z.
        Capable of processing multiple data on receiving vectorized input.
        """
        return self._blocktree.Findblock(x, y, z).astype(np.int64)
    
# --- FIND BLOCK AND CELL ID BY COORDINATE -------------------------------------
    def findblockcell(self, block, x, y, z):
        """
        Return the index of the blocks cell containing the coordinates x, y, z.
        Make sure that the provided block actually containes the coordinates.
        If the correct block index is unknown, findblock should be used in advance.
        Capable of processing multiple data on receiving vectorized input.
        """
        ix = ((x-self.__block_xmin[block])*self.__cell_xsize_recp[block]).astype(np.int32)
        iy = ((y-self.__block_ymin[block])*self.__cell_ysize_recp[block]).astype(np.int32)
        iz = ((z-self.__block_zmin[block])*self.__cell_zsize_recp[block]).astype(np.int32)
        i = np.clip(ix, 0, self.__nxb-1)
        j = np.clip(iy, 0, self.__nyb-1)
        k = np.clip(iz, 0, self.__nzb-1)
        return (k, j, i)
        
    def findcell(self, x, y, z):
        """
        Return the index of the block and the flat index of the blockcell
        containing the coordinates x, y, z.
        Capable of processing multiple data on receiving vectorized input.
        """
        block = self.findblock(x, y, z)
        kji = self.findblockcell(block, x, y, z)
        return block, kji

# --- FIND BLOCK AND FRACTIONAL CELL ID BY COORDINATE --------------------------
    def findblockfc(self, block, x, y, z):
        """
        Return the float index of the blocks cell containing the coordinates x, y, z.
        Make sure that the provided block actually containes the coordinates!
        If the correct block index is unknown, findblock should be used in advance.
        Capable of processing multiple data on receiving vectorized input.
        """
        fx = (x-self.__block_xmin[block])*self.__cell_xsize_recp[block] -.5
        fy = (y-self.__block_ymin[block])*self.__cell_ysize_recp[block] -.5
        fz = (z-self.__block_zmin[block])*self.__cell_zsize_recp[block] -.5
        return (fz, fy, fx)
        
    def findfc(self, x, y, z):
        """
        Return the index of the block and the float index of the blockcell
        containing the coordinates x, y, z.
        Capable of processing multiple data on receiving vectorized input.
        """
        block = self.findblock(x, y, z)
        return block, self.findblockfc(block, x, y, z)

# --- NEIGHBORHOOD DISCOVERY ---------------------------------------------------
    def findneigh(self, block, iz_bl, iy_bl, ix_bl, combine=False):
        """
        Implementation in progress; NOT YET CHECKED!
        """
        select_outskirt = np.array(np.meshgrid(*3*((1,0,0,1),))).sum(axis=0) > 0
        # Calculate coordinates of given cells:
        x_reloffset = np.linspace(-.5, .5, self.__nxb, endpoint=False) +.5/self.__nxb
        y_reloffset = np.linspace(-.5, .5, self.__nyb, endpoint=False) +.5/self.__nyb
        z_reloffset = np.linspace(-.5, .5, self.__nzb, endpoint=False) +.5/self.__nzb
        grid_reloffset = np.array(np.meshgrid(y_reloffset, z_reloffset, x_reloffset))
        reloffset = grid_reloffset[(2,0,1),:,:,:][:,iz_bl,iy_bl,ix_bl].T
        offset = self._bbox_size[block,:]*reloffset
        coords = offset +self._bbox_center[block,:]
        # Calculate coords of all potential neigborhood cells,
        # considering they may be refined by one level:
        cellsize = self.cellsize(block).T
        n_reloffset = np.array([-.75, -.25, .25, .75])
        grid_neigh_reloffset = np.array(np.meshgrid(*3*(n_reloffset,)))
        # neigh_reloffset = grid_neigh_reloffset[(2,0,1),:,:,:].reshape(3,-1)
        neigh_reloffset = grid_neigh_reloffset[(2,0,1),:,:,:][:,select_outskirt]
        neigh_offset = cellsize[:,:,None] *neigh_reloffset
        neigh_coords = coords[:,:,None] +neigh_offset
        #X, Y, Z = neigh_coords[:,0], neigh_coords[:,1], neigh_coords[:,2]        
        # Wrap neighborhood coordinates around cell boundaries
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = self.extent()
        X = modclip(neigh_coords[:,0], xmin, xmax)
        Y = modclip(neigh_coords[:,1], ymin, ymax)
        Z = modclip(neigh_coords[:,2], zmin, zmax)
        # Get blocks+cells from neighbor cell coordinates
        neigh_block, (neigh_iz, neigh_iy, neigh_ix) = self.findcell(X, Y, Z)
        # Remove doublettes; obviously, data has to be repacked to lists here,
        # because the number of neighbor doublettes per cell is not constant,
        # so the result will not be of regular format.
        if combine:
            N = set(zip(neigh_block.ravel(), neigh_iz.ravel(), neigh_iy.ravel(), neigh_ix.ravel()))
        else:
            result_iter = list(zip(neigh_block, neigh_iz, neigh_iy, neigh_ix))
            N = [set(zip(*cellneigh)) for cellneigh in result_iter]
        return deepcopy(N)
        
# --- SPATIAL BASEGRID ---------------------------------------------------------
    def extent(self, axis=None):
        xmin, ymin, zmin = self._domain_min
        xmax, ymax, zmax = self._domain_max
        extent = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])
        if axis is None:
            return extent
        elif axis in (0,1,2):
            ax_x, ax_y = {0:[1,2], 1:[0,2], 2:[0,1]}[axis]
            return np.array([extent[ax_x], extent[ax_y]])
        else:
            raise ValueError('Axis out of range.')
                    
    def resolution(self, axis=None):
        if axis is None:
            return self._domain_cells
        elif axis in (0,1,2):
            return self._domain_cells[axis]
        else:
            raise ValueError('Axis out of range.')
            
    def spacing(self, axis=None, res=None):
        if axis is None:
            if res is None:
                res = self._domain_cells
            if res[0] is None:
                res[0] = self._domain_cells[0]
            if res[1] is None:
                res[1] = self._domain_cells[1]
            if res[2] is None:
                res[2] = self._domain_cells[2]
            spacing = self._domain_size / res
        elif axis in (0,1,2):
            if res is None:
                res = self._domain_cells[axis]
            spacing = self._domain_size[axis]/res
        else:
            raise ValueError('Axis out of range.')
        return spacing
        
    def grating(self, axis, res=None, extent=None, align=None):
        """
        Partition the given axis of the given extend / entire simulation domain,
        equally into res parts (return the center coordinates of those parts).
        If no resolution is provided, the resolution will correspond to the
        best grid refinements resolution, with the coordinates in the centers
        of those grids cells.
        Typically used to construct grids for sampling points in the sim space.
        """
        if not axis in (0,1,2):
            raise ValueError('Axis out of range.')
        #
        if extent is None:
            extent_min = self._domain_min[axis]
            extent_max = self._domain_max[axis]
        else:
            extent_min, extent_max = extent
            
        if res is None:
            provisional_step = self.spacing(axis)
        else:
            provisional_step = (extent_max-extent_min) / res
            
        # If close to native res: align grating with cells
        if np.isclose(provisional_step, self.spacing(axis), rtol=1./7., atol=0.):
            align = True if align is None else bool(align)
            
        if align:
            grating_step = self.spacing(axis)
            grating_ref = self._domain_min[axis] +.5*grating_step
        else:
            grating_step = provisional_step
            extent_excess = (extent_max-extent_min) % grating_step
            grating_ref = extent_min +.5*grating_step +.5*extent_excess

        s = np.sign(extent_max-extent_min)
        ref_offset = extent_min-grating_ref
        ref_offset_steps = s*ceil(s*ref_offset/grating_step)
        grating_start = grating_ref +grating_step*ref_offset_steps
        grating_steps = floor((extent_max-extent_min) / grating_step)
        grating = grating_start +grating_step*np.arange(grating_steps) 
        return grating
            
    def basegrid_2d(self, axis, xres=None, yres=None, zres=None,
        x_extent=None, y_extent=None, z_extent=None):
        if axis == 0:
            z_grating = self.grating(2, zres, z_extent)
            y_grating = self.grating(1, yres, y_extent)
            return np.meshgrid(z_grating, y_grating)
        elif axis == 1:
            z_grating = self.grating(2, zres, z_extent)
            x_grating = self.grating(0, xres, x_extent)
            return np.meshgrid(z_grating, x_grating)
        elif axis == 2:
            y_grating = self.grating(1, yres, y_extent)
            x_grating = self.grating(0, xres, x_extent)
            return np.meshgrid(y_grating, x_grating)
        else:
            raise ValueError('Axis out of range.')

    def basegrid_3d(self, xres=None, yres=None, zres=None,
        x_extent=None, y_extent=None, z_extent=None):
        #
        y_grating = self.grating(1, yres, y_extent)
        x_grating = self.grating(0, xres, x_extent)
        z_grating = self.grating(2, zres, z_extent)
        return np.meshgrid(y_grating, x_grating, z_grating)

    def varblockgrating(self, axis):
        if axis not in (0,1,2):
            raise ValueError('Axis %i out of range(3).' % axis)
        blockbounds = np.concatenate((self._bbox_min[:,axis], self._bbox_max[:,axis]))
        II = (blockbounds-self._domain_min[axis])/self._bbox_minsize[axis]
        Ipartition = np.sort(list(set(np.around(II.T))))    
        IMid = .5*(Ipartition[1:]+Ipartition[:-1])
        IWidth = Ipartition[1:]-Ipartition[:-1]    
        grating = self._domain_min[axis] + IMid*self._bbox_minsize[axis]
        return grating, IWidth

    def varcellgrating(self, axis, extent=None):
        if axis not in (0,1,2):
            raise ValueError('Axis %i out of range(3).' % axis)
        self.__build_cell_coords()
        cellbounds = np.concatenate((
            self.__cellbox_min[:,axis].flatten(),
            self.__cellbox_max[:,axis].flatten(), ))
        II = (cellbounds-self._domain_min[axis])/self.spacing(axis=axis)
        Ipartition = np.sort(list(set(np.around(II.T))))    
        IMid = .5*(Ipartition[1:]+Ipartition[:-1])
        IWidth = Ipartition[1:]-Ipartition[:-1]    
        grating = self._domain_min[axis] + IMid*self.spacing(axis=axis)
        if extent is None:
            return grating, IWidth
        else:
            grating_min, grating_max = extent
            mask = (grating>=grating_min)*(grating<grating_max)
            return grating[mask], IWidth[mask]
            

pm_octree = pm3dgrid


################################################################################
##### CLASS: flash file 3d paramesh data grid ##################################
#####        provides guard cell and interpolation infrastructure ##############
################################################################################
# Prepare coefficients for trilinear interpolation
binmesh_y, binmesh_z, binmesh_x = np.meshgrid((0,1),(0,1),(0,1))
binmesh_xf = binmesh_x.flatten()
binmesh_yf = binmesh_y.flatten()
binmesh_zf = binmesh_z.flatten()
# Prepare coefficients for bilinear interpolation
binmesh_uf = np.array([0,0,0,0])
binmesh_vf = np.array([0,1,0,1])
binmesh_wf = np.array([0,0,1,1])

def floorpart(array):
    ia = np.floor(array).astype(np.int64)
    fa = array - ia
    return ia, fa

class datagrid(object):
    # gbld: guarded block data
    def __init__(self, blkdata, fgrid, ng=2, gbld_input=False, verbose=False):
        self._ng = max(int(ng),0)  ### Thickness of guard cell layer
        self.fgrid = fgrid
        self.octree = self.fgrid
        
        # Define some shorthands
        ng = self._ng
        rlevel = self.fgrid.blk_rlevel()
        ntype = self.fgrid.blk_ntype()
        
        # Prepare array shape for block data with guard cells:
        blocks, nk, nj, ni = blkdata.shape
        # Fill non-guard area with block data
        if ng>0 and not gbld_input:
            gbld_shape = (blocks, nk+2*ng, nj+2*ng, ni+2*ng)
            self.gbld = np.full(gbld_shape, 0., dtype=blkdata.dtype)
            self.gbld[:,ng:-ng,ng:-ng,ng:-ng] = blkdata
            if verbose:
                print('Filling guard cells...')
                sys.stdout.flush()
            for r in set(rlevel):
                blocks = np.where((rlevel==r)*(ntype==1))[0]
                chunks = 1+int(len(blocks)/1000)
                for i in range(chunks):
                    block_range = blocks[np.arange(i, len(blocks), chunks)]
                    X = self.fgrid.coords(block_range, 0, ng=ng)
                    Y = self.fgrid.coords(block_range, 1, ng=ng)
                    Z = self.fgrid.coords(block_range, 2, ng=ng)
                    self.gbld[block_range] = self.coeval(X, Y, Z,
                        interpolation='gfill', wrap=True)
        else:
            self.gbld = blkdata
        # Fill guard-cells
        # IDEA:
        # 1. Sort blocks by refinement level
        # 2. Work from low to high refinement by simply using the existing
        #    linear interpolation evaluation on the guard cell nodes,
        #    ignoring missing guard cells on those.
        # 3. TAKE CARE OF BOUNDARY WRAP AROUND!
        # This should work because:
        # A. Guard cells already filled if up-resolving
        # B. Guard cells not necessary for down-resolving,
        #    because outermost points are more outermost on higher rlevel
        
        if ng<2: return
        self.tricube = tricubic(self.gbld, verbose=True, ng=self._ng)
        
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False

    def blkdata(self, ng_out=0, force_ng=False):
        lk, lj, li = self.gbld.shape[-3:]
        if force_ng:
            dng = self._ng-ng_out
            if not dng in range(self._ng+1):
                raise ValueError('Requested amount of guard layers is invalid')
        else:
            dng = np.clip(self._ng-ng_out, 0, self._ng)
        return self.gbld[..., dng:lk-dng, dng:lj-dng, dng:li-dng]
        
    def gradient(self):
        """
        WARNING: UNTESTED CODE SECTION!
        """
        ng = self._ng
        order = np.clip(int(ng), 1, 2)
        xdiff, ydiff, zdiff = np.gradient(self.gbld, edge_order=order, axis=[-1,-2,-3])
        dx = self.fgrid.cellsize(axis=0)[...,None,None,None]
        dy = self.fgrid.cellsize(axis=1)[...,None,None,None]
        dz = self.fgrid.cellsize(axis=2)[...,None,None,None]
        xgrad = xdiff/dx
        ygrad = ydiff/dy
        zgrad = zdiff/dz
        lk, lj, li = self.gbld.shape[-3:]
        blkcore = [Ellipsis,slice(ng, lk-ng),slice(ng, lj-ng),slice(ng, li-ng)]
        return xgrad[blkcore], ygrad[blkcore], zgrad[blkcore]

    def derivative(self, axis=None):
        """
        WARNING: UNTESTED CODE SECTION!
        """
        if axis is None:
            return self.derivative(axis=0), self.derivative(axis=1), self.derivative(axis=2)
        elif not axis in (0,1,2):
            raise ValueError('Axis %i out of range(3).' % axis)
        ng = self._ng
        order = np.clip(int(ng), 1, 2)
        adiff = np.gradient(self.gbld, edge_order=order, axis=-1-axis)
        da = self.fgrid.cellsize(axis=axis)[...,None,None,None]
        agrad = adiff/da
        lk, lj, li = self.gbld.shape[-3:]
        blkcore = [Ellipsis,slice(ng, lk-ng),slice(ng, lj-ng),slice(ng, li-ng)]
        return agrad[blkcore]

# --- DATA AQUISITON -----------------------------------------------------------
    def coeval(self, X_in, Y_in, Z_in, interpolation=None, axis=None, wrap=True):
        """
        Evaluates the scalar block dataset at given positions.
        Capable of processing multiple data on receiving vectorized X,Y,Z input.
        This will typically be components of a grid.
        Caution: X,Y,Z need to have identical shapes;
        if using a 2d-grid, fill the remaining variable accordingly,
        which can be done by numpy.full_like.
        """
        ng = self._ng
        # Wrap coordinates to values inside grid domain:
        if np.all(wrap):
            ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = self.fgrid.extent()
            X = modclip(X_in, xmin, xmax)
            Y = modclip(Y_in, ymin, ymax)
            Z = modclip(Z_in, zmin, zmax)
        elif len(wrap)==3:
            X = modclip(X_in, xmin, xmax) if wrap[0] else X_in
            Y = modclip(Y_in, ymin, ymax) if wrap[1] else Y_in
            Z = modclip(Z_in, zmin, zmax) if wrap[2] else Z_in
        else:
            X = X_in
            Y = Y_in
            Z = Z_in
        #
        if interpolation is None or interpolation == 'none':
            # Do no interpolation, but directly use data from cell
            # containing the provided coordinates.
            #block, cell = self.fgrid.findcell(X,Y,Z)
            #return self.blkdata().reshape(flatcell_shape)[block,cell]
            block, (iz, iy, ix) = self.fgrid.findcell(X,Y,Z)
            return self.gbld[block, iz+ng, iy+ng, ix+ng]
            
        elif interpolation == 'cubic' or interpolation == 'tricubic':
            if ng<2:
                return self.coeval(X_in, Y_in, Z_in, interpolation='bilinear',
                                   axis=axis, wrap=wrap)
            block, fc = self.fgrid.findfc(X,Y,Z)
            return self.tricube.eval(block, fc)

        elif interpolation == 'bilinear' or interpolation == 'linear':
            # !!! CHECK IF PERMUTATIONS CORRECT !!!
            if ng<1:
                return self.coeval(X_in, Y_in, Z_in, interpolation='none', wrap=wrap)
            elif axis is None:
                return self.coeval(X_in, Y_in, Z_in, interpolation='trilinear', wrap=wrap)
            block, (fcz, fcy, fcx) = self.fgrid.findfc(X,Y,Z)
            # Calculate floor cell indicies
            # Temporarily shift values up to evade rounding dirswitch at 0.
            icx, dcx = floorpart(fcx)
            icy, dcy = floorpart(fcy)
            icz, dcz = floorpart(fcz)
            # Get all cell index combinations of (floor,floor+1) over (ax-2, ax-1)
            icu = (icx, icy, icz)[axis]
            icv = (icx, icy, icz)[axis-2]
            icw = (icx, icy, icz)[axis-1]
            mesh_icu = icu[Ellipsis, None] + binmesh_uf + ng
            mesh_icv = icv[Ellipsis, None] + binmesh_vf + ng
            mesh_icw = icw[Ellipsis, None] + binmesh_wf + ng
            # Calculate interpolation coefficients for (floor,floor+1) over x,y,z
            dcv = (dcx, dcy, dcz)[axis-2]
            dcw = (dcx, dcy, dcz)[axis-1]
            stackshape = [Ellipsis, None, None]
            av = np.concatenate((1.-dcv[stackshape], dcv[stackshape]), -1)
            aw = np.concatenate((1.-dcw[stackshape], dcw[stackshape]), -2)
            mesh_a = (av*aw).reshape(fcx.shape+(-1,))            
            # Apply interpolation coefficients and sum
            mesh_blk = np.stack(4*(block,), -1)
            mesh_icx = (mesh_icu, mesh_icv, mesh_icw)[-axis]
            mesh_icy = (mesh_icu, mesh_icv, mesh_icw)[-axis+1]
            mesh_icz = (mesh_icu, mesh_icv, mesh_icw)[-axis+2]
            mesh_data = self.gbld[mesh_blk, mesh_icz, mesh_icy, mesh_icx]
            return np.sum(mesh_a*mesh_data, axis=-1)

        elif interpolation == 'trilinear':
            if ng<1:
                return self.coeval(X_in, Y_in, Z_in, interpolation='none', wrap=wrap)
            block, (fcz, fcy, fcx) = self.fgrid.findfc(X,Y,Z)
            # Calculate floor cell indicies
            # Temporarily shift values up to evade rounding dirswitch at 0.
            icx, dcx = floorpart(fcx)
            icy, dcy = floorpart(fcy)
            icz, dcz = floorpart(fcz)
            # Get all cell index combinations of (floor,floor+1) over x,y,z
            mesh_icx = icx[Ellipsis, None] + binmesh_xf + ng
            mesh_icy = icy[Ellipsis, None] + binmesh_yf + ng
            mesh_icz = icz[Ellipsis, None] + binmesh_zf + ng
            # Calculate interpolation coefficients for (floor,floor+1) over x,y,z
            stackshape = [Ellipsis, None, None, None]
            ax = np.concatenate((1.-dcx[stackshape], dcx[stackshape]), -1)
            ay = np.concatenate((1.-dcy[stackshape], dcy[stackshape]), -2)
            az = np.concatenate((1.-dcz[stackshape], dcz[stackshape]), -3)
            mesh_a = (ax*ay*az).reshape(fcx.shape+(-1,))            
            # Apply interpolation coefficients and sum
            mesh_blk = np.stack(8*(block,), -1)
            mesh_data = self.gbld[mesh_blk, mesh_icz, mesh_icy, mesh_icx]
            return np.sum(mesh_a*mesh_data, axis=-1)
            
        elif interpolation == 'gfill':
            block, (fcz, fcy, fcx) = self.fgrid.findfc(X,Y,Z)
            np.round(fcz, 2, fcz)
            np.round(fcy, 2, fcy)
            np.round(fcx, 2, fcx)
            # Calculate floor cell indicies
            # Temporarily shift values up to evade rounding dirswitch at 0.
            icx = np.floor(fcx).astype(np.int32)
            icy = np.floor(fcy).astype(np.int32)
            icz = np.floor(fcz).astype(np.int32)
            # Get all cell index combinations of (floor,floor+1) over x,y,z
            mesh_icx = np.expand_dims(icx, -1) + binmesh_xf + ng
            mesh_icy = np.expand_dims(icy, -1) + binmesh_yf + ng
            mesh_icz = np.expand_dims(icz, -1) + binmesh_zf + ng
            # Calculate cell indicies float offset
            stackshape = fcx.shape+(1,1,1)
            dcx = np.reshape(fcx-icx, stackshape)
            dcy = np.reshape(fcy-icy, stackshape)
            dcz = np.reshape(fcz-icz, stackshape)
            # Calculate interpolation coefficients for (floor,floor+1) over x,y,z
            ax = np.concatenate((1.-dcx, dcx), -1)
            ay = np.concatenate((1.-dcy, dcy), -2)
            az = np.concatenate((1.-dcz, dcz), -3)
            mesh_a = (ax*ay*az).reshape(fcx.shape+(-1,))            
            # Apply interpolation coefficients and sum
            mesh_blk = np.stack(8*(block,), -1)
            try:
                mesh_data = self.gbld[mesh_blk, mesh_icz, mesh_icy, mesh_icx]
            except IndexError:
                print("EE: OUT OF BOUNDS:")
                print('shape: %s'%(self.gbld.shape,))
                print(mesh_blk.max(), mesh_icz.max(), mesh_icy.max(), mesh_icx.max())
                sys.stdout.flush()
                raise
            return np.sum(mesh_a*mesh_data, axis=-1)
            
        else:
            # Interpolation parameter setting unrecognized
            print('Warning: Unrecognized interpolation mode \'%s\''%(interpolation,))
            print('using \'none\' instead.')
            return self.eval(X, Y, Z, interpolation='none', wrap=wrap)
                        
    def eval(self, X, Y, Z, interpolation=None, axis=None, wrap=True,
        chunksize=100000, verbose=False):
        """
        Evaluates the scalar block dataset at given positions.
        This is done by spoonfeeding chunks of max. chunksize positions
        to coeval to limit the temporary memory usage of it.
        """
        result = np.full_like(X.ravel(), np.nan, dtype=self.gbld.dtype)
        baseshape = X.shape
        chunks = 1+int(len(result)/chunksize)
        for i in range(chunks):
            chunkresult = self.coeval(
                X.ravel()[i::chunks], Y.ravel()[i::chunks], Z.ravel()[i::chunks],
                interpolation=interpolation, axis=axis, wrap=wrap)
            result[i::chunks] = chunkresult
        return result.reshape(baseshape)

# --- DATA AQUISITON HELPERS FOR COMMON TASKS ----------------------------------        
    def read_slice(self, axis, offset=0.,
        interpolation='linear', xres=None, yres=None, zres=None,
        x_extent=None, y_extent=None, z_extent=None, wrap=True, verbose=False):
        """
        Reads a slice perpendicular to axis from the given flashfile data
        at the given axis intercept,
        e.g. axis=0 -> X, intercept=4e+19 => read slice at X=4e+19.
        If no resolution is provided, the finest native grid resolution
        of the actual grid of the flashfile is assumed.
        The extent of the processed area can be limited by providing
        e.g. x_extent=(x_min, x_max) for any axis.
        TODO: At this stage, the resolution refers to the entire unlimited domain.
        """
        if axis==0:
            Z, Y = self.fgrid.basegrid_2d(axis, xres=xres, yres=yres, zres=zres,
                x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
            X = np.full_like(Z, offset)
        elif axis==1:
            Z, X = self.fgrid.basegrid_2d(axis, xres=xres, yres=yres, zres=zres,
                x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
            Y = np.full_like(Z, offset)
        elif axis==2:
            Y, X = self.fgrid.basegrid_2d(axis, xres=xres, yres=yres, zres=zres,
                x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
            Z = np.full_like(Y, offset)
        else:
            raise ValueError('Axis out of range(3).' % axis)
        return self.eval(X, Y, Z, interpolation=interpolation, axis=axis,
                         wrap=wrap, verbose=verbose)
                    
    def equalize_blockaxis(self, axis, weight_grid):
        if axis not in (0,1,2):
            raise ValueError('Axis %i out of range(3).' % axis)
        ng = self._ng
        
        gbld_axis = -(axis+1)
        # Select the core area of the integration axis:
        l_axis = self.gbld.shape[gbld_axis]
        axis_core = slice(ng, l_axis-ng)
        gbld_core = [Ellipsis]+3*[slice(None)]
        gbld_core[gbld_axis] = axis_core

        data_gbld = (self.gbld)[tuple(gbld_core)]
        weight_gbld = (weight_grid.gbld)[tuple(gbld_core)]
        ones_gbld = np.ones_like(self.gbld)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            weights_eq = np.expand_dims(
                weight_gbld.mean(axis=gbld_axis), gbld_axis  ) * ones_gbld
            data_eq = np.expand_dims(
                (data_gbld*weight_gbld).mean(axis=gbld_axis), gbld_axis  ) * ones_gbld / weights_eq
        deq_grid = datagrid(np.nan_to_num(data_eq), self.fgrid, ng=ng, gbld_input=True)
        weq_grid = datagrid(np.nan_to_num(weights_eq), self.fgrid, ng=ng, gbld_input=True)
        return deq_grid, weq_grid
        
    def read_meanslice(self, axis, weight_data=None,
        interpolation='linear', xres=None, yres=None, zres=None, verbose=False,
        x_extent=None, y_extent=None, z_extent=None):
        """
        Reads a slice of weighted mean values, calculated along the given axis.
        from the given flashfile data and weights from flashfile data.
        If no weights data is provided, equal weights are assumed.
        If no resolution is provided, the finest native grid resolution
        of the actual grid of the flashfile is assumed.
        The resolution parameter of the (integrated) axis sets an upper
        limit for the depth scanning resolution.
        The extent of the processed area can be limited by providing
        e.g. x_extent=(x_min, x_max) for any axis.
        """
        if axis==0:
            Z, Y = self.fgrid.basegrid_2d(axis, xres=xres, yres=yres, zres=zres,
                x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
            X = None
        elif axis==1:
            Z, X = self.fgrid.basegrid_2d(axis, xres=xres, yres=yres, zres=zres,
                x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
            Y = None
        elif axis==2:
            Y, X = self.fgrid.basegrid_2d(axis, xres=xres, yres=yres, zres=zres,
                x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
            Z = None
        else:
            raise ValueError('Axis %i out of range(3).' % axis)
        
        axis_extent = (x_extent, y_extent, z_extent)[axis]
        if axis_extent is None:
            axis_extent = (-np.inf, np.inf)
        axmin, axmax = axis_extent
        coords = self.fgrid.coords()
        select_extent = (coords[:,axis]>axmin)*(coords[:,axis]<axmax)
        if weight_data is None:
            weight_data = 1.*select_extent
        else:
            weight_data *= select_extent
        
        weight_grid = datagrid(weight_data, self.fgrid, ng=self._ng)
                                
        # Accelerate grid scanning (1):
        # Equalize cell data in blocks along depth axis,
        # so that the depth scanning has to resolve the smallest block length only
        # instead of the smallest cell length:
        deq_g, weq_g = self.equalize_blockaxis(axis, weight_grid)
        #deq_g, weq_g = self, weight_grid
        
        # Accelerate grid scanning (2):
        # Instead scanning the full depth axis using the smallest block size,
        # scan only the midpoints of the finest resolution blocks 
        # covering the axis. (weigh the results accordingly)
        grating, gratingweights = self.fgrid.varblockgrating(axis)
        #grating = self.fgrid.grating(axis)
        #gratingweights = np.ones_like(grating)

        if verbose:
            answer  = str(len(grating))
            answer += ('*'+str(self.fgrid._nb[axis]))
            answer += ' / ' +str(self.fgrid._domain_blocks[axis]*self.fgrid._nb[axis]) +' (var)'
            print('Depth resolution: %s'%(answer,))
            print('Map resolution: %s'%(len((X,Y,Z)[axis-1])))
            sys.stdout.flush()
            
        result_accu = np.zeros_like((X,Y,Z)[axis-1])   
        weight_accu = np.zeros_like((X,Y,Z)[axis-1])
        
        for intercept, iweight in zip(grating, gratingweights):
            if axis==0:
                X = np.full_like(Z, intercept)
            elif axis==1:
                Y = np.full_like(Z, intercept)
            elif axis==2:
                Z = np.full_like(Y, intercept)
            dataslice = deq_g.eval(X, Y, Z, interpolation=interpolation,
                                   axis=axis, wrap=False)
            weightslice = weq_g.eval(X, Y, Z, interpolation=interpolation,
                                     axis=axis, wrap=False)
            result_accu += dataslice*weightslice*iweight
            weight_accu += weightslice*iweight
        return result_accu/weight_accu
            
    def read(self, xres=None, yres=None, zres=None,
        x_extent=None, y_extent=None, z_extent=None, interpolation=None,
        verbose=True):
        #
        Z, Y = self.fgrid.basegrid_2d(0, xres=xres, yres=yres, zres=zres,
            x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
        x_grating = self.fgrid.grating(0, xres, x_extent)
        #
        grid_data = np.full(x_grating.shape+Z.shape, np.nan, dtype=self.gbld.dtype)
        if verbose:
            print('Grid Map Resolution: %s'%(str(grid_data.shape,)))
        #
        Ni = len(x_grating)
        for i,x in enumerate(x_grating):
            X = np.full_like(Z, x)
            grid_data[i] = self.coeval(X, Y, Z, interpolation=interpolation)
        #print('\rEvaluated %i slices.        '%Ni)
        return grid_data

    def read_column(self, axis, x=0., y=0., z=0.,
        interpolation=None, xres=None, yres=None, zres=None):
        if axis not in (0,1,2):
            raise ValueError('Axis %i out of range(3).' % axis)
        axisres = (xres, yres, zres)[axis]
        grating = self.fgrid.grating(axis, axisres)
        if axis==0:
            Y = np.full_like(grating, y)
            Z = np.full_like(grating, z)
            data_line = self.eval(grating, Y, Z,
                interpolation=interpolation)
        elif axis==1:
            X = np.full_like(grating, x)
            Z = np.full_like(grating, z)
            data_line = self.eval(X, grating, Z,
                interpolation=interpolation)
        else: # axis==2
            X = np.full_like(grating, x)
            Y = np.full_like(grating, y)
            data_line = self.eval(X, Y, grating,
                interpolation=interpolation)
        return grating, data_line
        
    def read_meancolumn(self, axis, weight_data=None, interpolation='linear',
        x_extent=None, y_extent=None, z_extent=None):
        #
        if axis not in (0,1,2):
            raise ValueError('Axis %i out of range(3).' % axis)
        #
        axis_grating, noweights = self.fgrid.varcellgrating(axis)
        #
        ax0, ax1, ax2 = axis, [0,1,2][axis-2], [0,1,2][axis-1]
        #
        ax0_extent = (x_extent, y_extent, z_extent)[axis]
        if ax0_extent is not None:
            axmin, axmax = ax0_extent
            select_extent = (axis_grating>axmin)*(axis_grating<axmax)
            ax0_grating = axis_grating[select_extent]
        else:
            ax0_grating = axis_grating
        #
        
        coords = self.fgrid.coords()
        plane_extent = (coords[:,ax0]==coords[:,ax0])
        #
        ax1_extent = (x_extent, y_extent, z_extent)[ax1]
        if ax1_extent is not None:
            ax1min, ax1max = ax1_extent
            plane_extent *= (coords[:,ax1]>ax1min)*(coords[:,ax1]<ax1min)
        
        ax2_extent = (x_extent, y_extent, z_extent)[ax2]
        if ax2_extent is not None:
            ax2min, ax2max = ax2_extent
            plane_extent *= (coords[:,ax2]>ax2min)*(coords[:,ax2]<ax2max)
        
        if weight_data is None:
            weight_data = 1.*plane_extent
        else:
            weight_data *= plane_extent
        #
        weight_grid = datagrid(weight_data, self.fgrid, ng=self._ng)
        
        # Accelerate grid sampling:
        # Equalize cell data in blocks along both integration axis,
        # so that the grid sampling has to resolve the smallest block length only
        # instead of the smallest cell length:
        deq1_g, weq1_g = self.equalize_blockaxis(ax1, weight_grid)
        deq_g, weq_g = deq1_g.equalize_blockaxis(ax2, weq1_g)
        
        XYZ = [None, None, None]
        ax1_grating, ax1_iweights = self.fgrid.varblockgrating(ax1)
        ax2_grating, ax2_iweights = self.fgrid.varblockgrating(ax2)
        XYZ[ax1], XYZ[ax2] = np.meshgrid(ax1_grating, ax2_grating)
        Wax1, Wax2 = np.meshgrid(ax1_iweights, ax2_iweights)
        W = Wax1*Wax2
        #W /= W.sum()

        result_line = np.zeros_like(ax0_grating)
        #
        for i, intercept in enumerate(ax0_grating):
            XYZ[ax0] = np.full_like(W, intercept)
            dataslice = deq_g.eval(XYZ[0], XYZ[1], XYZ[2], interpolation=interpolation)
            weightslice = weq_g.eval(XYZ[0], XYZ[1], XYZ[2], interpolation=interpolation)            
            #valueslice = dataslice*weightslice/weightslice.sum()
            #result_line[i] = np.sum(valueslice*W)
            result_line[i] = np.average(dataslice, weights=W*weightslice)
        return ax0_grating, result_line


def divergence(datagrid_x, datagrid_y, datagrid_z):
    div_x = datagrid_x.derivative(axis=0)
    div_y = datagrid_y.derivative(axis=1)
    div_z = datagrid_z.derivative(axis=2)
    return div_x +div_y +div_z

# ==== TEST SETTINGS ===========================================================  
filename = 'M:/scratchbin/MW_supermuc/CF97M/E.B100N1/CF97TE_hdf5_plt_cnt_0160'

# ==== TEST ====================================================================    
if __name__ == '__main__':
    pc = 3.0856776e+18     # parsec (cm)
    ffile = plotfile(filename, memorize=False)
    ffile.learn(var_mhd)
    ffile.learn(var_ch5)
    ffile.learn(var_grid)
    fgrid = pm3dgrid(ffile)
    
    #dd = fgrid.coords(axis=2)/pc
    dd = ffile['temp']
    wd = ffile['numdens']
    ffile.close()

    ddgrid = datagrid(dd, fgrid, verbose=True) # NOTE: guard cell count ng=0 (OFF)

    axis = 2    
    offset = 1e-5*pc
    #Si = ddgrid.read_slice(axis, offset, interpolation='linear', verbose=True)
    Mi = ddgrid.read_meanslice(axis, interpolation='linear', weight_data=wd,
                               verbose=True, xres=2852, yres=713, zres=713)
    
    #import cProfile
    #import pstats
    #print '#############################################################'
    #cProfile.run("Mi = ddgrid.read_meanslice(axis, interpolation='linear', verbose=True)", 'restats')
    #p = pstats.Stats('restats')
    #p.strip_dirs().sort_stats('time').print_stats(15)
    #p.strip_dirs().sort_stats('cumtime').print_stats(15)
