# ==============================================================================
#  2-D FLASH TRACER INTEGRATION
#  Author: Benedikt Weinhold  (~Sep/Oct 2025)
#
#  SNAPSHOT_2D
#
#  Classstrcture to read in and store data for simulation snapshots from FLASH
#  (implemented) or other codes (have to implement yourself).
#  Can be used as a template to use the same code for other simulation outputs
#  See what functionalities Snapshot2D has to have in README 
# ==============================================================================

from abc import ABC, abstractmethod
import numpy as np
import h5py as h5
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import shared_memory


class Snapshot2DBase(ABC):
    """
    Abstract base class for 2D simulation snapshots.
    """

    # Required abstract methods / attributes

    #exact simulation time at snapshot 
    @property
    @abstractmethod
    def simtime(self):
        pass

    #dictionary for all thermodynamic variables ['dens', 'ye', ...] {key: array}
    @property
    @abstractmethod
    def td_vars(self):
        pass


    #The central function to implement - gives the values of a given variable in td_vars interpolated from output to exact x,y (z) position
    @abstractmethod
    def getQuantAtPos(self, td_var, x, y,):
        pass

    # --- Shared memory methods implemented here ---
    @classmethod
    def from_shm(cls, shm_registry_entry):
        """
        Reconstruct a Snapshot2D-like object from shared memory buffers
        created by create_shm_for_snapshot_generic().
        Keeps SharedMemory handles alive to prevent premature release.
        """
        self = object.__new__(cls)  # bypass __init__

        def parse_dtype(dtype_meta):
            if isinstance(dtype_meta, str):
                try:
                    return np.dtype(eval(dtype_meta))
                except Exception:
                    return np.dtype(dtype_meta)
            else:
                return np.dtype(dtype_meta)

        # Store SharedMemory handles to prevent GC
        self._shm_handles = []

        # Iterate over all attributes in the SHM metadata
        for attr_name, attr_meta in shm_registry_entry.items():
            if attr_name == "td_vars":
                continue  # handle separately

            if isinstance(attr_meta, dict) and "name" in attr_meta:
                shm = shared_memory.SharedMemory(name=attr_meta["name"])
                self._shm_handles.append(shm)
                arr = np.ndarray(attr_meta["shape"], parse_dtype(attr_meta["dtype"]), buffer=shm.buf)
                setattr(self, attr_name, arr)
            else:
                # scalar or metadata-only
                setattr(self, attr_name, attr_meta.get("value", None))

        # Handle td_vars separately
        self._td_vars = {}
        td_vars_meta = shm_registry_entry.get("td_vars", {})
        for key, meta in td_vars_meta.items():
            shm = shared_memory.SharedMemory(name=meta["name"])
            self._shm_handles.append(shm)
            self._td_vars[key] = np.ndarray(meta["shape"], parse_dtype(meta["dtype"]), buffer=shm.buf)

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


class Snapshot2DFLASH(Snapshot2DBase):

    def __init__(self, path, keys, only_leafs=True):
        with h5.File(path, 'r') as f:
            if only_leafs:
                node_type = f['node type'][:] 
                leaf_mask = (node_type == 1)
            else:
                leaf_mask = slice(None)

            # Bounding boxes and scalar fields
            self._bbox = f['bounding box'][:][leaf_mask]

            #idea: read in raw data - with weird h5 fmt - make dict out of it, store wanted quantity as attribute: self.attr
            # then only store attributes (normal scalars and arrays in shm to reconstruct from)
            realscalars_raw = f['real scalars'][:]
            realscalars_dict = {
                key.decode('utf-8').strip() if isinstance(key, bytes) else str(key): value
                for key, value in realscalars_raw
            }
            self._simtime = realscalars_dict["time"]

            integerscalars_raw = f['integer scalars'][:]
            integerscalars_dict = {
                key.decode('utf-8').strip() if isinstance(key, bytes) else str(key): value
                for key, value in integerscalars_raw
            }
            self._cells_per_block_x = integerscalars_dict['nxb'] 
            self._cells_per_block_y = integerscalars_dict['nyb']

            realruntimeparams_raw = f['real runtime parameters'][:]
            realruntimedata_dict = {
                key.decode('utf-8').strip() if isinstance(key, bytes) else str(key): value
                for key, value in realruntimeparams_raw
            }
            self.xmin = realruntimedata_dict['xmin']
            self.xmax = realruntimedata_dict['xmax']
            self.ymin = realruntimedata_dict['ymin']
            self.ymax = realruntimedata_dict['ymax']

            # Thermodynamic & neutrino quantities
            self._td_vars = {}
            for key in keys:
                self._td_vars[key] = np.squeeze(f[key][:][leaf_mask])
            

        self._precompute_cell_coords()

    # Properties required by abstract base class
    
    @property
    def simtime(self):
        return self._simtime
    
    @property
    def td_vars(self):
        return self._td_vars

    def _precompute_cell_coords(self):
        """Precompute cell centers and edges for all blocks (vectorized)."""
        n_blocks = self._bbox.shape[0]
        
        # Extract all bounding boxes at once
        xmin = self._bbox[:, 0, 0]  # shape: (n_blocks,)
        xmax = self._bbox[:, 0, 1]
        ymin = self._bbox[:, 1, 0]
        ymax = self._bbox[:, 1, 1]
        
        # Compute deltas for all blocks
        delta_x = (xmax - xmin) / self._cells_per_block_x  # shape: (n_blocks,)
        delta_y = np.abs(ymax - ymin) / self._cells_per_block_y
        
        # Create cell index arrays
        cell_indices_x = np.arange(self._cells_per_block_x)  # shape: (cells_per_block_x,)
        cell_indices_y = np.arange(self._cells_per_block_y)
        
        # Vectorized computation for all blocks
        # Broadcasting: (n_blocks, 1) + (1, cells_per_block) = (n_blocks, cells_per_block)
        self.cell_centers_x = xmin[:, np.newaxis] + (cell_indices_x + 0.5) * delta_x[:, np.newaxis]
        
        y_lower = np.minimum(ymin, ymax)  # Ensure ascending order
        self.cell_centers_y = y_lower[:, np.newaxis] + (cell_indices_y + 0.5) * delta_y[:, np.newaxis]
        
        # Cell edges
        self.x_low = xmin[:, np.newaxis] + cell_indices_x * delta_x[:, np.newaxis]
        self.x_high = self.x_low + delta_x[:, np.newaxis]
        self.y_low = ymin[:, np.newaxis] + cell_indices_y * delta_y[:, np.newaxis]
        self.y_high = self.y_low + delta_y[:, np.newaxis]

    def cellCoords(self, blockID, tr_id=None, with_edges=False):
        """Return precomputed cell coordinates for a block."""
        if with_edges:
            return (self.cell_centers_x[blockID], self.cell_centers_y[blockID],
                    self.x_low[blockID], self.x_high[blockID],
                    self.y_low[blockID], self.y_high[blockID])
        return self.cell_centers_x[blockID], self.cell_centers_y[blockID]

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
        return self.simtime
    
    def getSimArea(self):
        return self.xmin, self.xmax, self.ymin, self.ymax

    def findBlock(self, x, y, tr_id): #tr_id for testing if singular tracer does weird thing

        x_mask = (self._bbox[:,0,0] <= x) & (x < self._bbox[:,0,1])
        y_mask = (self._bbox[:,1,0] <= y) & (y < self._bbox[:,1,1])

        block_ids = np.where(x_mask & y_mask)[0]

        if len(block_ids) == 1:
            return block_ids[0]
        elif len(block_ids) == 0:
            print(f"⚠️ Tracer {tr_id}: No block found for point ({x}, {y})", flush=True)
            return None
        else:
            #if Only_leafs = False the last found index is normally the leaf block
            return block_ids[len(block_ids)-1]

    def interpolateBlock(self, var, blockID, tr_id):
        x_coords, y_coords = self.cellCoords(blockID, tr_id)
        return RegularGridInterpolator((x_coords, y_coords),
                                       self.td_vars[var][blockID].squeeze().T,
                                       method='linear',
                                       bounds_error=False,
                                       fill_value=None)

    def getQuantAtPos(self, var, x, y, tr_id=None):

        xmin, xmax, ymin, ymax = self.getSimArea()

        eps_inner = 1       #1cm
        eps_outer = 1e3     #10m

        out_of_bounds = (
            (x < xmin + eps_inner) or (x > xmax - eps_outer) or
            (y < ymin + eps_outer) or (y > ymax - eps_outer)
        )

        if out_of_bounds and tr_id is not None:
            #print(f'tracer {tr_id} accesses {var} OOB at: ({x:.8e}, {y:.8e}), returning 0')
            return 0

        blockID = self.findBlock(x,y, tr_id)
        interp_func = self.interpolateBlock(var, blockID, tr_id)
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

    def get_unbound_composition(self):
        """
        Go through all cells that are unbound (ener + gpot > 0 and vrad > 0)
        and sum up/average the composition.

        Returns
        -------
        ejected_mass : float
            Total mass of all unbound cells.
        massfractions : dict
            Dictionary of isotopes with their mass fraction in the ejecta:
            { 'h1': {'Z':1, 'A':1, 'X': 0.00168}, ... }
        """
        ejected_mass = 0.0
        massfractions = {iso: 0.0 for iso in self.isotopes}

        # Loop over blocks
        for block_id in range(len(self.bbox)):
            # Cell centers
            x_ccenters, y_ccenters = self.cellCoords(block_id)
            x_ccenters_2d = x_ccenters[:, None] * np.ones((1, self.cells_per_block_y))
            y_ccenters_2d = np.ones((self.cells_per_block_x, 1)) * y_ccenters[None, :]

            # Cell properties
            cvols = self.cellVolumes(block_id)
            cdens = self.td_vars['dens'][block_id]
            cvelx = self.td_vars['velx'][block_id]
            cvely = self.td_vars['vely'][block_id]
            cener = self.td_vars['ener'][block_id]
            cgpot = self.td_vars['gpot'][block_id]

            # Compute radial velocity
            r = np.sqrt(x_ccenters_2d**2 + y_ccenters_2d**2)
            dot = x_ccenters_2d * cvelx + y_ccenters_2d * cvely
            vrad = dot / np.maximum(r, 1e-20)

            # Mask for unbound cells
            mask = (cener + cgpot > 0) & (vrad > 0)

            # Mass of each cell
            cmasses = cdens * cvols

            # Total ejected mass in this block
            ejected_mass_block = np.sum(cmasses[mask])
            ejected_mass += ejected_mass_block

            # Sum mass contribution of each isotope
            for iso in self.isotopes:
                iso_masses = self.isotopes[iso][block_id] * cmasses
                mask_flat = mask.flatten()
                iso_masses_flat = iso_masses.flatten()
                massfractions[iso] += np.sum(iso_masses_flat[mask_flat])

        # Convert absolute isotope masses to mass fractions
        if ejected_mass > 0:
            for iso in massfractions:
                massfractions[iso] = {
                    'Z': self.rn_16[iso]['Z'],
                    'A': self.rn_16[iso]['A'],
                    'X': massfractions[iso] / ejected_mass
                }
        else:
            # No ejecta; return zero fractions
            for iso in massfractions:
                massfractions[iso] = {
                    'Z': self.rn_16[iso]['Z'],
                    'A': self.rn_16[iso]['A'],
                    'X': 0.0
                }

        return ejected_mass, massfractions


# test_HeS_930 = Snapshot2DFLASH('/home/guest/Bene/HeS_2D/HeS_net/HeS_rn16e_hdf5_plt_cnt_0930', ['temp'])

# print(test_HeS_930.getQuantAtPos('temp', 0.31e9, -0.25e9)/1e9)

