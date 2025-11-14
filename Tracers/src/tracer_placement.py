# ==============================================================================
#  2-D FLASH TRACER INTEGRATION
#  Author: Benedikt Weinhold  (~Sep 2025)
#
#  TRACER_PLACEMENT
#
#  Functions to place tracers in the simulation
#  For further explanations on the parameters see README.md 
# ==============================================================================

from time import time
import numpy as np
from astropy.constants import M_sun
import random


from .utils import write_log 
from config import PATH_TO_OUTPUT, ONLY_UNBOUND, MAX_DENS_PLACE, MAX_TEMP_PLACE, YE_STEPS

# -------------- POSITION PROPORTIONAL TO DENSITY -----------------------------

def PosFromDens_blockbased(snap, ptN):
    """
    Place tracer particles in a FLASH snapshot, favoring regions of higher density
    and areas with significant Ye deviations. Returns the positions and masses of tracers.
    
    Parameters
    ----------
    snap : Snapshot2D
        snapshot object containing cell data.
    ptN : int
        Total number of tracers to place.
    
    Returns
    -------
    ptX, ptY, ptM : np.ndarray
        Arrays of tracer x-positions, y-positions, and masses.
    """

    t0 = time()

    # Lists to store info per block
    block_masses = []
    block_cell_info = []
    block_cell_ye = []

    # Helper to decide which cells are eligible for tracer placement
    def compute_mask(cdens, cvelx, cvely, cener, cgpot, ctemp):
        r = np.sqrt(x_ccenters**2 + y_ccenters**2)
        dot = x_ccenters * cvelx + y_ccenters * cvely
        cvrad = dot / np.maximum(r, 1e-20)

        # Only keep cells that meet energy/temp criteria
        if ONLY_UNBOUND:
            mask = (cener + cgpot > 0) & (cvrad > 0) & (ctemp < MAX_TEMP_PLACE)
        else:
            mask = (cdens < MAX_DENS_PLACE) & (ctemp < MAX_TEMP_PLACE)
        return mask

    # Loop over all blocks to collect relevant cell info
    for block_id in range(len(snap.bbox)):
        # Get cell positions and edges
        x_ccenters, y_ccenters, x_clows, x_chighs, y_clows, y_chighs = snap.cellCoords(block_id, with_edges=True)
        
        # Get cell properties
        cvols = snap.cellVolumes(block_id)
        cdens = snap.td_vars['dens'][block_id]
        cvelx = snap.td_vars['velx'][block_id]
        cvely = snap.td_vars['vely'][block_id]
        cener = snap.td_vars['ener'][block_id]
        cgpot = snap.td_vars['gpot'][block_id]
        cye = snap.td_vars['ye'][block_id]
        ctemp = snap.td_vars['temp'][block_id]

        # Compute mass per cell
        cmasses = cvols * cdens

        # Determine which cells are eligible
        mask = compute_mask(cdens, cvelx, cvely, cener, cgpot, ctemp)

        # Flatten arrays to simplify selection
        cmasses_flat = cmasses.flatten()
        cye_flat = cye.flatten()
        xlow_flat = np.tile(x_clows, snap.cells_per_block_y)
        xhigh_flat = np.tile(x_chighs, snap.cells_per_block_y)
        ylow_flat = np.repeat(y_clows, snap.cells_per_block_x)
        yhigh_flat = np.repeat(y_chighs, snap.cells_per_block_x)
        mask_flat = mask.flatten()

        # Select only valid cells
        masses = cmasses_flat[mask_flat]
        ye = cye_flat[mask_flat]
        x_low = xlow_flat[mask_flat]
        x_high = xhigh_flat[mask_flat]
        y_low = ylow_flat[mask_flat]
        y_high = yhigh_flat[mask_flat]

        # Store block info
        total_block_mass = masses.sum()
        block_masses.append(total_block_mass)
        block_cell_info.append((masses, x_low, x_high, y_low, y_high))
        block_cell_ye.append(ye)

    # Log total mass selected
    M_sun_g = M_sun.value * 1e3
    total_mass = sum(block_masses)
    write_log(PATH_TO_OUTPUT, f"Total selected mass: {np.round(total_mass / M_sun_g, 3)} M_sun")

    # Determine number of tracers per block
    tracers_per_block = []
    approx_mass_p_tr = total_mass / ptN
    for bm in block_masses:
        ntr = max(int(round(bm / approx_mass_p_tr)), 1)  # ensure at least 1 tracer per block
        tracers_per_block.append(ntr)

    # Place tracers inside blocks
    ptX, ptY, ptM = [], [], []

    for block_id, ntr in enumerate(tracers_per_block):
        masses, x_low, x_high, y_low, y_high = block_cell_info[block_id]
        if len(masses) == 0:
            continue  # skip blocks with no valid cells

        # Increase tracer density in regions where Ye deviates from 0.5
        if YE_STEPS:
            ye_dev = np.abs(np.array(block_cell_ye[block_id]) - 0.5).max()
            if ye_dev > 0.06:
                ntr *= 16
            elif ye_dev > 0.04:
                ntr *= 8
            elif ye_dev > 0.02:
                ntr *= 4

        # Compute sampling probabilities based on cell mass
        probs = masses / masses.sum()
        tracer_mass = block_masses[block_id] / ntr

        # Randomly pick cells and assign tracer positions and mass
        chosen_idx = np.random.choice(len(masses), size=ntr, p=probs)
        for idx in chosen_idx:
            ptX.append(random.uniform(x_low[idx], x_high[idx]))
            ptY.append(random.uniform(y_low[idx], y_high[idx]))
            ptM.append(tracer_mass)

    # Log total number of tracers placed
    write_log(PATH_TO_OUTPUT, f"Placed {len(ptX)} tracers in {(time()-t0):.2f}s")

    return np.array(ptX), np.array(ptY), np.array(ptM)



# ------------- PLACE AT SPECIFIC LOCATIONS --------------------------------

def PosFromFile(path_to_datfile):
    """
    Read tracer positions and masses from an external ASCII file.
    
    Parameters
    ----------
    path_to_datfile : str
        Path to the tracer data file.
    
    Returns
    -------
    ptX, ptY, ptM : np.ndarray
        Arrays of tracer x-positions, y-positions, and masses.
    """

    # Load tracer data from file (at least 2D for consistent handling)
    tr_pos = np.atleast_2d(np.genfromtxt(path_to_datfile))

    # Extract columns for x, y positions and mass
    ptX = tr_pos[:, 0]
    ptY = tr_pos[:, 1]
    ptM = tr_pos[:, 2]

    # Return as separate arrays
    return ptX, ptY, ptM