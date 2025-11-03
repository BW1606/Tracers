# ==============================================================================
#  2-D FLASH TRACER INTEGRATION
#  Author: Benedikt Weinhold  (~Sep/Oct 2025)
#
#  SNAPSHOT_2D
#
#  Class to read in and store data for simulation snapshots from FLASH
#  Can be used as a template to use the same code for other simulation outputs
#  See what functionalities Snapshot2D has to have in README 
# ==============================================================================


import numpy as np
import h5py as h5
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import shared_memory

#while testing: 
import sys
from glob import glob
from time import time

"""
TODO's:

-once tracer program sufficiently tested, rm tr_id arguments

"""
class Snapshot2D:
        # constructor pre 'with_rn16'
        # with h5.File(path, 'r') as f:
        #     if only_leafs:
        #         node_type = f['node type'][:] 
        #         leaf_mask = (node_type == 1)
        #         bbox = f['bounding box'][:][leaf_mask]
        #         realscalars = f['real scalars'][:]
        #         realruntimeparams = f['real runtime parameters'][:]
        #         td_vars = {}
        #         for key in keys:
        #             td_vars[key] = np.squeeze(f[key][:][leaf_mask])
        #     else:
        #         bbox = f['bounding box'][:]
        #         realscalars = f['real scalars'][:]
        #         realruntimeparams = f['real runtime parameters'][:]
        #         td_vars = {}
        #         for key in keys:
        #             td_vars[key] = np.squeeze(f[key][:])

        # self.bbox = bbox
        # self.realruntimeparams = realruntimeparams
        # self.realscalars = realscalars
        # self.td_vars = td_vars

    def __init__(self, path, keys, only_leafs=True, with_rn16=False):
        with h5.File(path, 'r') as f:
            if only_leafs:
                node_type = f['node type'][:] 
                leaf_mask = (node_type == 1)
            else:
                leaf_mask = slice(None)

            # Bounding boxes and scalar fields
            self.bbox = f['bounding box'][:][leaf_mask]
            self.realscalars = f['real scalars'][:]
            self.realruntimeparams = f['real runtime parameters'][:]

            # Time-dependent variables
            self.td_vars = {}
            for key in keys:
                self.td_vars[key] = np.squeeze(f[key][:][leaf_mask])

            # Optional: read reduced network 16 composition
            if with_rn16:
                self.rn_16 = {
                    'h1':   {'Z': 1, 'A': 1},
                    'he4':  {'Z': 2, 'A': 4},
                    'c12':  {'Z': 6, 'A': 12},
                    'o16':  {'Z': 8, 'A': 16},
                    'ne20': {'Z': 10,'A': 20},
                    'mg24': {'Z': 12,'A': 24},
                    'si28': {'Z': 14,'A': 28},
                    's32':  {'Z': 16,'A': 32},
                    'ar36': {'Z': 18,'A': 36},
                    'ca40': {'Z': 20,'A': 40},
                    'ti44': {'Z': 22,'A': 44},
                    'cr48': {'Z': 24,'A': 48},
                    'fe52': {'Z': 26,'A': 52},
                    'fe54': {'Z': 26,'A': 54},
                    'ni56': {'Z': 28,'A': 56},
                    'neut': {'Z': 0, 'A': 1},
                }
                self.isotopes = {}
                for iso in self.rn_16:
                    if iso in f:
                        self.isotopes[iso] = f[iso][:][leaf_mask]
                    else:
                        # fill with zeros if isotope not present in snapshot
                        self.isotopes[iso] = np.zeros(np.sum(leaf_mask))


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

    def getQuantAtPos(self, var, x, y, tr_id=None):

        xmin, xmax, ymin, ymax = self.getSimArea()

        #sometimes  x goes through this OOB check but then doesnt find a block in findBlock() - floating point uncertainty?
        eps = 1e-10
        if x-eps < xmin or x+eps > xmax or y-eps < ymin or y+eps > ymax:
            if tr_id is not None:
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

    @classmethod
    def from_shm(cls, shm_registry_entry):
        """
        Reconstruct a Snapshot2D-like object from shared memory buffers
        and keep SharedMemory handles alive to prevent premature release.
        """
        self = object.__new__(cls)  # bypass __init__

        def parse_dtype(dtype_meta):
            """
            Convert stored dtype string or object back to np.dtype.
            Handles structured dtypes like "[('name','S80'),('value','<f8')]"
            """
            if isinstance(dtype_meta, str):
                try:
                    # Try evaluating structured dtype
                    return np.dtype(eval(dtype_meta))
                except Exception:
                    # fallback to simple dtype
                    return np.dtype(dtype_meta)
            else:
                return np.dtype(dtype_meta)

        # Store SharedMemory handles to prevent garbage collection
        self._shm_handles = []

        # Attach bbox, realscalars, realruntimeparams
        for attr in ["bbox", "realscalars", "realruntimeparams"]:
            meta = shm_registry_entry[attr]
            shm = shared_memory.SharedMemory(name=meta["name"])
            self._shm_handles.append(shm)
            setattr(
                self,
                attr,
                np.ndarray(meta["shape"], parse_dtype(meta["dtype"]), buffer=shm.buf)
            )

        # Attach td_vars
        self.td_vars = {}
        for key, meta in shm_registry_entry["td_vars"].items():
            shm = shared_memory.SharedMemory(name=meta["name"])
            self._shm_handles.append(shm)
            self.td_vars[key] = np.ndarray(meta["shape"],  parse_dtype(meta["dtype"]), buffer=shm.buf)

        # static metadata
        self.cells_per_block_x = shm_registry_entry.get("cells_per_block_x", 16)
        self.cells_per_block_y = shm_registry_entry.get("cells_per_block_y", 16)

        return self

    def close_shm(self):
        """Close all attached shared memory blocks."""
        if hasattr(self, "_shm_handles"):
            for shm in self._shm_handles:
                try:
                    shm.close()
                except Exception:
                    pass
            self._shm_handles = [] 

    def get_unbound_composition(self, maxTemp=10e9):
        """
        Go through all cells that are unbound (ener + gpot > 0 and vrad > 0)
        and sum up/average the composition.
        Uses the same masking as PosFromDens_Bene_steps().
        """
        ejected_mass = 0.0
        massfractions = {iso: 0.0 for iso in self.isotopes}

        # Loop over blocks
        for block_id in range(len(self.bbox)):
            # Cell centers
            x_ccenters, y_ccenters = self.cellCoords(block_id)
            
            # Broadcast to 2D arrays for cell-by-cell operations
            x_ccenters_2d = np.tile(x_ccenters[:, None], (1, self.cells_per_block_y))
            y_ccenters_2d = np.tile(y_ccenters[None, :], (self.cells_per_block_x, 1))

            # Cell properties
            cvols = self.cellVolumes(block_id)
            cdens = self.td_vars['dens'][block_id]
            cvelx = self.td_vars['velx'][block_id]
            cvely = self.td_vars['vely'][block_id]
            cener = self.td_vars['ener'][block_id]
            cgpot = self.td_vars['gpot'][block_id]
            ctemp = self.td_vars['temp'][block_id]

            # Compute radial velocity
            r = np.sqrt(x_ccenters_2d**2 + y_ccenters_2d**2)
            vrad = (x_ccenters_2d * cvelx + y_ccenters_2d * cvely) / np.maximum(r, 1e-20)

            # Mask for unbound cells (same as placement method)
            mask = (cener + cgpot > 0) & (vrad > 0) & (ctemp < maxTemp)

            # Mass of each cell
            cmasses = cdens * cvols

            # Total ejected mass in this block
            ejected_mass += np.sum(cmasses[mask])

            # Sum mass contribution of each isotope
            for iso in self.isotopes:
                iso_masses = self.isotopes[iso][block_id][0].squeeze() * cmasses
                massfractions[iso] += np.sum(iso_masses[mask])

        # Convert absolute isotope masses to mass fractions
        if ejected_mass > 0:
            for iso in massfractions:
                massfractions[iso] = {
                    'Z': self.rn_16[iso]['Z'],
                    'A': self.rn_16[iso]['A'],
                    'X': massfractions[iso] / ejected_mass
                }
        else:
            for iso in massfractions:
                massfractions[iso] = {
                    'Z': self.rn_16[iso]['Z'],
                    'A': self.rn_16[iso]['A'],
                    'X': 0.0
                }

        return ejected_mass, massfractions







# path_to_pltfiles = '/home/bweinhold/Auswertung/2D_Analysis/nucleosynthesis_analysis/HeS_s13.8/HeS_rn16e_hdf5_plt_cnt_0792'
# plt_files = path_to_pltfiles#sorted(glob(path_to_pltfiles + "*", recursive=False))

# testsnap = Snapshot2D(plt_files, ['temp', 'dens', 'ye', 'ener', 'gpot', "velx", 'vely'], only_leafs=False, with_rn16 = True)

# unboundmass, massfracs = testsnap.get_unbound_composition()
# print(unboundmass)
