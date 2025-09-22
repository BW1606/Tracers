import numpy as np
import h5py as h5
from scipy.interpolate import RegularGridInterpolator
from glob import glob
import sys
from time import time



"""
TODO's:

-once tracer program sufficiently tested, rm tr_id arguments

"""
class Snapshot2D:

    def __init__(self, path, keys, only_leafs = True):

        with h5.File(path, 'r') as f:
            if only_leafs:
                node_type = f['node type'][:] 
                leaf_mask = (node_type == 1)
                bbox = f['bounding box'][:][leaf_mask]
                realscalars = f['real scalars'][:]
                realruntimeparams = f['real runtime parameters'][:]
                td_vars = {}
                for key in keys:
                    td_vars[key] = np.squeeze(f[key][:][leaf_mask])
            else:
                bbox = f['bounding box'][:]
                realscalars = f['real scalars'][:]
                realruntimeparams = f['real runtime parameters'][:]
                td_vars = {}
                for key in keys:
                    td_vars[key] = np.squeeze(f[key][:])

        self.bbox = bbox
        self.realruntimeparams = realruntimeparams
        self.realscalars = realscalars
        self.td_vars = td_vars
        self.cells_per_block_x = 16 #TODO: read in from hdf5 file itself
        self.cells_per_block_y = 16

    def cellCoords(self, blockID, with_edges = False):
        xmin = self.bbox[blockID, 0, 0]
        xmax = self.bbox[blockID, 0, 1]
        ymin = self.bbox[blockID, 1, 0]
        ymax = self.bbox[blockID, 1, 1]

        delta_x = (xmax - xmin) / self.cells_per_block_x
        delta_y = np.abs(ymax - ymin) / self.cells_per_block_y

        cell_centers_x = xmin + (np.arange(self.cells_per_block_x) + 0.5) * delta_x
        y_lower = np.minimum(ymin, ymax)  # Ensure ascending order
        cell_centers_y = y_lower + (np.arange(self.cells_per_block_y) + 0.5) * delta_y

        if with_edges:
            x_low = xmin + (np.arange(self.cells_per_block_x)) * delta_x
            x_high = xmin + (np.arange(self.cells_per_block_x)) * delta_x + delta_x
            y_low = ymin + (np.arange(self.cells_per_block_y)) * delta_y
            y_high = ymin + (np.arange(self.cells_per_block_y)) * delta_y + delta_y
            return cell_centers_x, cell_centers_y,x_low, x_high, y_low, y_high

        return cell_centers_x, cell_centers_y

  
    def cellVolumes(self, blockID):
        """
        Compute the volume of each cell in a block for 2D axisymmetric FLASH sims.
        Formula: V = pi * (r_outer^2 - r_inner^2) * dy
        Returns a (Ny, Nx) array of cell volumes.
        """
        # Block bounds
        xmin = np.squeeze(self.bbox[blockID, 0, 0])
        xmax = np.squeeze(self.bbox[blockID, 0, 1])
        ymin = np.squeeze(self.bbox[blockID, 1, 0])
        ymax = np.squeeze(self.bbox[blockID, 1, 1])

        # Cell spacing
        dx = (xmax - xmin) / self.cells_per_block_x
        dy = (ymax - ymin) / self.cells_per_block_y

        # Radial cell edges (Nx+1)
        x_edges = np.linspace(xmin, xmax, self.cells_per_block_x + 1)

        # Compute radial contribution: pi * (r_out^2 - r_in^2)
        shell_areas = np.pi * (x_edges[1:]**2 - x_edges[:-1]**2)  # shape (Nx,)

        # Multiply by dy for each row (broadcast over y)
        vol_block = np.outer(np.ones(self.cells_per_block_y), shell_areas) * dy

        return vol_block

    def currentSimTime(self):
        return self.realscalars[2][1]
    
    def getSimArea(self):
        xmin = self.realruntimeparams[130][1]
        xmax = self.realruntimeparams[129][1]
        ymin = self.realruntimeparams[133][1]
        ymax = self.realruntimeparams[132][1]
        return xmin, xmax, ymin, ymax

    def findBlock(self, x, y, tr_id): #tr_id for testing if singular tracer does weird thing

        x_mask = (self.bbox[:,0,0] <= x) & (x < self.bbox[:,0,1])
        y_mask = (self.bbox[:,1,0] <= y) & (y < self.bbox[:,1,1])

        block_ids = np.where(x_mask & y_mask)[0]

        if len(block_ids) == 1:
            return block_ids[0]
        elif len(block_ids) == 0:
            print(f"⚠️ Tracer {tr_id}: No block found for point ({x}, {y})", flush=True)
            return None
        else:
            return block_ids[len(block_ids)-1]

    def interpolateBlock(self, var, blockID):
        x_coords, y_coords = self.cellCoords(blockID)
        return RegularGridInterpolator((x_coords, y_coords), self.td_vars[var][blockID].squeeze().T, method='linear', bounds_error=False, fill_value=None) #TODO: zwischen 0-0.5 dx checken, method = nearest

    def getQuantAtPos(self, var, x, y, tr_id):

        xmin, xmax, ymin, ymax = self.getSimArea()

        #sometimes  x goes through this OOB check but then doesnt find a block in findBlock() - floating point uncertainty?
        eps = 1e-10
        if x-eps < xmin or x+eps > xmax or y-eps < ymin or y+eps > ymax:
            print(f'tracer {tr_id} accesses {var} OOB at: {x,y}, returning 0')
            return 0

        # if x-eps < xmin:
        #     x = xmin+eps

        # if x+eps > xmax:
        #     x = xmax-eps

        # if y-eps < ymin:
        #     y = ymin+eps

        # if y+eps > ymax:
        #     y = ymax-eps

        blockID = self.findBlock(x,y, tr_id)
        interp_func = self.interpolateBlock(var, blockID)
        val = interp_func([x,y]).item()

        if not np.isfinite(val):
            print(f"tracer {tr_id} accesses {var} but its nan: {val}", flush=True)

        return val

    def findCell(self, x, y):
        
        blockID= self.findBlock(x=x, y=y, tr_id=1)

        xc_centers, yc_centers = self.cellCoords(blockID)

        x_id = np.abs(xc_centers - x).argmin()
        y_id = np.abs(yc_centers - y).argmin()

        return blockID, x_id, y_id

    def enclosedMass(self):
        """
        Compute enclosed mass as a function of radius for the 2D axisymmetric snapshot.
    
        Returns
        -------
        r_sorted : 1D array
            Radii of all cells, sorted in ascending order.
        M_enc : 1D array
            Corresponding cumulative enclosed mass at each radius.
        """
        all_r = []
        all_m = []
    
        # Loop over blocks
        for block_id in range(len(self.bbox)):
            x_cells, y_cells = self.cellCoords(block_id)
            vol = self.cellVolumes(block_id)
            dens = self.td_vars["dens"][block_id]
    
            # Flatten arrays
            x_flat = x_cells[:, None] * np.ones((1, self.cells_per_block_y))
            y_flat = np.ones((self.cells_per_block_x, 1)) * y_cells[None, :]
            r = np.sqrt(x_flat**2 + y_flat**2)
            mass_cell = dens * vol
    
            all_r.append(r.flatten())
            all_m.append(mass_cell.flatten())
    
        all_r = np.concatenate(all_r)
        all_m = np.concatenate(all_m)
    
        # Sort by radius
        sort_idx = np.argsort(all_r)
        r_sorted = all_r[sort_idx]
        M_enc = np.cumsum(all_m[sort_idx])
    
        return r_sorted, M_enc



