import numpy as np
import sys
from .tricubic_direct import tricubic

__author__ = "Michael Weis"
__version__ = "1.0.0.0"

################################################################################
# ==== HELPER ==================================================================
################################################################################

def modclip(a, a_min, a_max):
    return a_min + np.mod(a-a_min, a_max-a_min)


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
        which can be done by e.g. Z=numpy.full_like(X, z_offset)
        for an X,Y grid at a fixed z-offset.
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
    def get_slice_coords(self, axis, offset=0., **bgparam):
        X, Y, Z = self.fgrid.basegrid_2d(axis, **bgparam)
        if axis==0:
            X = np.full_like(Z, offset)
        elif axis==1:
            Y = np.full_like(Z, offset)
        elif axis==2:
            Z = np.full_like(Y, offset)
        else:
            raise ValueError('Axis out of range(3).')
        return X, Y, Z        
    
    def read_slice(self, axis, offset=0.,
        interpolation='linear', wrap=True, verbose=False, **bgparam):
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
        #bgparam = dict(xres=xres, yres=yres, zres=zres,
        #    x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
        X, Y, Z = self.get_slice_coords(axis, offset=offset, **bgparam)
        return self.eval(X, Y, Z, interpolation=interpolation, axis=axis,
                         wrap=wrap, verbose=verbose)
                    
    def equalize_blockaxis(self, axis, weight_grid):
        if axis not in (0,1,2):
            raise ValueError('Axis out of range(3).')
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
        
    def extent_coverage(self, axis, x_extent, y_extent, z_extent):
        axis_extent = (x_extent, y_extent, z_extent)[axis]
        if axis_extent is None:
            axis_extent = (-np.inf, np.inf)
        extent_min, extent_max = axis_extent
        cellcenter = self.fgrid.coords()[:,axis]
        cellsize   = self.fgrid.cellsize()[axis][:,None,None,None]
        cell_min = cellcenter -.5*cellsize
        cell_extent_relmin = (extent_min-cell_min)/cellsize
        cell_extent_relmax = (extent_max-cell_min)/cellsize
        cell_extent_fmin = np.clip(cell_extent_relmin, 0., 1.)
        cell_extent_fmax = np.clip(cell_extent_relmax, 0., 1.)
        cell_coverage = cell_extent_fmax - cell_extent_fmin
        return cell_coverage

    def integrate_axis(self, axis, interpolation='linear',
            xres=None, yres=None, zres=None,
            x_extent=None, y_extent=None, z_extent=None):
            
        # XXX TODO BAUSTELLE TODO XXX
        X, Y, Z = self.fgrid.basegrid_2d(axis, xres=xres, yres=yres, zres=zres,
                x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
        
        depth_coverage = self.extent_coverage(axis, x_extent, y_extent, z_extent)  
        weight_grid = datagrid(depth_coverage, self.fgrid, ng=self._ng)
                                
        # Accelerate grid scanning (1):
        # Equalize cell data in blocks along depth axis,
        # so that the depth scanning has to resolve the smallest block length only
        # instead of the smallest cell length:
        deq_g, weq_g = self.equalize_blockaxis(axis, weight_grid)
        
        # Accelerate grid scanning (2):
        # Instead scanning the full depth axis using the smallest block size,
        # scan only the midpoints of the finest resolution blocks 
        # covering the axis. (weigh the results accordingly)
        grating, gratedepth = self.fgrid.varblockgrating(axis)
        
        # Get number of cells per block along the given axis
        nax = self.fgrid._nb[axis]

        if verbose:
            answer  = str(len(grating))
            answer += ('*'+str(self.fgrid._nb[axis]))
            answer += ' / ' +str(self.fgrid._domain_blocks[axis]*nax) +' (var)'
            print(f'Depth resolution: {answer}')
            mapres = len((X,Y,Z)[axis-1])
            print(f'Map resolution: {mapres}')
            sys.stdout.flush()
            
        coord_shape = (X,Y,Z)[axis-1].shape
        result_accu = np.zeros(coord_shape)
        
        for intercept, dL in zip(grating, gratedepth):
            if axis==0:
                X = np.full(coord_shape, intercept)
            elif axis==1:
                Y = np.full(coord_shape, intercept)
            elif axis==2:
                Z = np.full(coord_shape, intercept)
            blockax_avgdata_slice = deq_g.eval(X, Y, Z, interpolation=interpolation,
                                   axis=axis, wrap=False)
            blockax_coverage_slice = weq_g.eval(X, Y, Z, interpolation=interpolation,
                                     axis=axis, wrap=False)
            blockax_len_slice = nax*dL * blockax_coverage_slice
            result_accu += blockax_avgdata_slice * blockax_len_slice
        return result_accu
        
    def average_axis(self, axis, weight_data=None,
            interpolation='linear', verbose=False,
            xres=None, yres=None, zres=None,
            x_extent=None, y_extent=None, z_extent=None):
        """
        Reads a slice of weighted average values, calculated along the given axis.
        from the given flashfile data and weights from flashfile data.
        If no weights data is provided, equal weights are assumed.
        If no resolution is provided, the finest native grid resolution
        of the actual grid of the flashfile is assumed.
        The resolution parameter of the (integrated) axis sets an upper
        limit for the depth scanning resolution.
        The extent of the processed area can be limited by providing
        e.g. x_extent=(x_min, x_max) for any axis.
        """
        X, Y, Z = self.fgrid.basegrid_2d(axis, xres=xres, yres=yres, zres=zres,
                x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
        
        depth_coverage = self.extent_coverage(axis, x_extent, y_extent, z_extent)
        
        if weight_data is None:
            weight_data = depth_coverage
        else:
            weight_data *= depth_coverage
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
        Xfoo, Y, Z = self.fgrid.basegrid_2d(0, xres=xres, yres=yres, zres=zres,
                x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
        #
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


################################################################################
##### CLASS: weighted flash file 3d paramesh data grid #########################
# XXX CAUTION: THIS IS AN EXPERIMENTAL CONSTRUCTION SITE! XXX #
################################################################################
'''
class wdgrid(object):
    # gbld: guarded block data (with or without inner nodes)
    # gblw: guarded block data weights (same format as gbld)
    def __init__(self, blkdata, blkweights, octree, ng=2,
            guard_input=False, verbose=False):
        ## Save essential data
        self._ng = ng  # Thickness of guard cell layer
        self.octree = octree
        ## Define some shorthands
        rlevel = self.fgrid.blk_rlevel()
        ntype = self.fgrid.blk_ntype()
        # Prepare array shape for block data with guard cells:
        blocks, nk, nj, ni = blkdata.shape
        # Check validity of guard cell layer thickness
        if 2*ng > min(nk,nj,ni) or ng < 0:
            raise RuntimeError(f'Invalid thickness ({ng}) of guard cell layer.')
        ## Fill guarded block data
        blkdw = blkdata*blkweights
        if ng == 0:
            self.gblw  = np.copy(blkweights)
            self.gbldw = blkdw
        else:
            ## Create placeholder fields to be filled with guarded data
            gbld_shape = (blocks, nk+2*ng, nj+2*ng, ni+2*ng)
            self.gblw  = np.full(gbld_shape, 0., dtype=blkdw.dtype)
            self.gbldw = np.full(gbld_shape, 0., dtype=blkdw.dtype)
            ## Fill non-guarded data
            self.gblw[:,ng:-ng,ng:-ng,ng:-ng]  = blkweights
            self.gbldw[:,ng:-ng,ng:-ng,ng:-ng] = blkdw
            ## Fill leaf block (guard cells)
            for r in set(rlevel):
                rlevel_mask = rlevel==r
                leaf_mask = ntype==1
                blocks = np.where(rlevel_mask*leaf_mask)[0]
                
                chunks = 1+int(len(blocks)//1000)
                for i in range(chunks):
                    block_range = blocks[np.arange(i, len(blocks), chunks)]
                    X = octree.coords(block_range, 0, ng=ng)
                    Y = octree.coords(block_range, 1, ng=ng)
                    Z = octree.coords(block_range, 2, ng=ng)
                    self.gbld[block_range] = self.coeval(X, Y, Z,
                        interpolation='gfill', wrap=True)
'''

