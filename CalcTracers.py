#default package dependencies
import numpy as np
from glob import glob
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from time import time
import os
import sys
import multiprocessing as mp
from datetime import datetime
import shutil
from decimal import Decimal
from scipy.interpolate import interp1d
import random
from astropy.constants import M_sun
import traceback

#self-programmed stuff - more info in the README.md
import Snapshot2dFLASH as Snap
import Progenitors as Prog

# -------------------- TODO's --------------------

"""
Things to still test/do now:
    - go through TODO's in the Code
    - finish the README.md
    - dont place tracers in some sleeve, cone around rotation axis?

Things that could be useful in the future:
    - run same trajectory with Max Jacobis Tracer Code --> Wrote mail 03/09
    - parallelize tracer file creation: (300s for 10k tracers atm) - is cpus or disk speed the problem ?
    - properly comment (maybe copy into AI see what happens)
    - (use shared memory buffer for snapshots instead of copying each to every CPU: see Tracers Max Jacobis Github)
    - save t_step_last(chunk) of tracer to use as first step in next chunk
    - instead of in the end, call seed calc in single tracer integration at start (fwd)/ end (bwd) chunk (def CalcSeed(r))
    - if all tracers not calc anymore -> terminate whole program, should never happen in full calc
"""

# -------------------- CONFIG --------------------

#----------------WHERE ARE THE SNAPSHOTS-----------

path_to_pltfiles = 'path/to/simulation_output_snapshots'
plt_files = sorted(glob(path_to_pltfiles + "*", recursive=False))

#--- WHERE SHOULD THE TRACERS GO / OUTPUT DIRECTORY ----------

path_to_tracers = f'/path/to/output/dir'

#------ WHAT INFO DO THE TRACERS NEED TO HAVE -----------

with_neutrinos = True #False or True, see readme/somewhere else

#--------CALCULATE ONLY UNTIL NSE?-----------------

only_until_NSE = False
max_temp = 6e9 #11GK - tracers that reach 10e9K (well above NSE) will not be calculated further in the next chunks (only if backward)
#TODO: NSE flag temp and calc_until max temp are different


#--------CALCULATE INITIAL COMPOSITION--------------

calc_seeds = True #True or False, if True then after tracer integration the code calculates the initial composition of the tracer

prog_type = 'FLASH' #where the progenitor file comes from. i.e. which progenitor class gets called


path_to_progfile = '../2D_simulation/HeS_s13.8/HeS_s13.8.1d'
#path_to_progfile = '../Progenitors/S15_NuGrid_log423.data'

#---------- INTEGRATION DIRECTION ---------------

direction = 'backward'  # backward or forward

# -------- TRACER PLACEMENT ---------------------

# Further info in README and my thesis
placement_method = 'PosWithDens' # 'PosWithDens' or 'FromFile'

#if placement_method = 'FromFile' where from:
path_to_tracer_start = '../data/test_positions/test_positions_bw_2.dat'

#if placement_method = 'PosWithDens':
num_tracers = 10000
only_unbound = True
max_dens = 1e11 #only needed if only_unbound = False (generally 1e11, tested by Max Witt)


# -------- ADDITIONAL MESSAGE TO LOG FILE ------------------

arb_message = f''       

# -------------TECHNICALITIES -------------------------

chunk_size = 50
rtol = 1e-2            #new standart up until now:
atol = 1e4             
maxstep = 1e-4         #1e-4#np.nan   
PER_TRACER_TIMEOUT = 60.0 #max time [s] a tracer can take for chunk integration, otherwise the tracer is terminated and an exception is raised

# -------------------- DONT TOUCH ---------------------------

solve_ivp_args = [rtol, atol, maxstep]
xmin, xmax, ymin, ymax = 0, 3.2e9, -3.2e9, 3.2e9 #FLASH standard values?


#----------------------- UTILITIES -----------------------

def is_ejected(gpot, ener, velx, vely, x, y):
    """
    Determines if particles are ejected based on energy criteria.
    
    A particle is considered ejected if it has:
    1. Positive radial velocity (moving outward)
    2. Positive total energy (kinetic + potential > 0)
    
    Parameters:
    -----------
    gpot : array_like
        Gravitational potential energy values for particles
    ener : array_like  
        Total energy values for particles (kinetic + internal + etc.)
    velx : array_like
        x-component of velocity for particles
    vely : array_like
        y-component of velocity for particles
    x : array_like
        x-position coordinates of particles
    y : array_like
        y-position coordinates of particles
        
    Returns:
    --------
    ejected : ndarray
        Binary array where 1 indicates ejected particles and 0 indicates bound particles
    """
    # Calculate radial distance from origin
    r = np.sqrt(x**2 + y**2)
    
    # Avoid division by zero by setting minimum radius
    r_safe = np.maximum(r, 1e-30)
    
    # Calculate radial velocity component (dot product of velocity with radial unit vector)
    v_r = (velx * x + vely * y) / r_safe
    
    # Determine ejection criteria:
    # 1. Radial velocity positive (moving outward)
    # 2. Total energy (ener + gpot) positive (unbound from gravitational potential)
    ejected = ((v_r > 0) & ((ener + gpot) > 0)).astype(int)
    
    return ejected

def write_log(output_dir, msg):
    """
    Writes a timestamped message to a log file in the specified directory.
    
    Creates or appends to a log file with consistent formatting, ensuring
    the output directory exists before writing. Useful for tracking
    script execution and debugging processes.
    
    Parameters:
    -----------
    output_dir : str
        Directory path where the log file will be created/stored
    msg : str
        Message content to be written to the log file
        
    Returns:
    --------
    None
    """
    logfile_path=os.path.join(output_dir, 'run.log')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Ensure the directory for the logfile exists
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)

    with open(logfile_path, 'a') as logfile:
        logfile.write(f'[{timestamp}] {msg}\n')
        logfile.flush()

def L_from_f(f, r):
    """
    Calculates luminosity from local flux and radius using the inverse-square law.
    
    Converts flux measurements to luminosity by accounting for the
    spherical surface area through which the flux is distributed.
    
    Parameters:
    -----------
    f : array_like
        Flux values (energy per unit time per unit area)
    r : array_like
        Radius values (distance from source)
        
    Returns:
    --------
    L : array_like
        Luminosity values (total energy per unit time)
    """
    return 4*np.pi*r**2*f

def write_to_tracer_file(times, positions, keys, data_dict, tracer_entries, tr_id, output_dir, tracer_entries_fmt):
    
    """
    Write tracer quantities to a .dat file. Complicated since quantities in the tracer file arent the same as the snapshot keys

    Parameters
    ----------
    times : array-like
        Time steps (n_times,)
    positions : tuple of arrays
        (x, y) arrays of shape (n_times,)
    data_dict : dict
        {key: np.array(shape=(n_times,))}
    keys : list
        Quantities available in data_dict.
    tracer_entries : list
        Quantities to write.
    tr_id : int
        Tracer ID.
    output_dir : str
        Directory for output file.
    """

    x, y = positions
    r = np.sqrt(x**2 + y**2)

    out = {}

    for entry in tracer_entries:
        if entry in keys:
            if entry == 'temp':
                out[entry] = data_dict[entry]/1e9 #WinNet wants GK
            else:
                out[entry] = data_dict[entry]

        elif entry == 'r':
            out[entry] = r/1e5 #cm to km

        elif entry in ['lnue', 'lnua']:
            flux_key = 'f' + entry[1:]   # 'lnue' -> 'fnue'
            out[entry] = L_from_f(data_dict[flux_key], r)*1e51 #flash give flux in B/s cm**2, WinNet needs erg/s

        elif entry in ['lnux', 'lanux']:
            if 'fnux' not in data_dict:
                raise KeyError("Missing flux fnux for lnux/lanux")
            Lx = L_from_f(data_dict['fnux'], r) *1e51 #flash give flux in B/s cm**2, WinNet needs erg/s
            out['lnux'] = 0.5 * Lx #Winnet expects heavy type x neutrinos in two kinds: anux and nux
            out['lanux'] = 0.5 * Lx

        elif entry == 'eanux':
            if 'enux' not in data_dict:
                raise KeyError("Missing enux for eanux")
            out['eanux'] = data_dict['enux']

        elif entry == 'ejected':
            out['ejected'] = is_ejected(
                data_dict['gpot'], data_dict['ener'],
                data_dict['velx'], data_dict['vely'], x, y
            )

        else:
            raise ValueError(f"Don't know how to handle tracer entry {entry}")

    # --- collect arrays for output ---
    chunk_tracer_dat = [times, x, y]  # always include these base quantities
    for entry in tracer_entries:
        if entry in out:
            chunk_tracer_dat.append(out[entry])

    # stack into (n_times, n_cols)
    chunk_tracer_dat_array = np.column_stack(chunk_tracer_dat)

    # --- formats ---
    entry_fmt_list = [tracer_entries_fmt[entry] for entry in ['t','x','y'] + tracer_entries]

    # --- write file ---
    filename = os.path.join(output_dir, f'tracer{str(tr_id).zfill(5)}.dat')
    with open(filename, 'a') as f:
        np.savetxt(f, chunk_tracer_dat_array, fmt=entry_fmt_list, comments='')

def ensure_ascending_time_order(path_to_tracers):
    """
    Ensures all tracer files have data ordered by ascending time.
    
    Processes tracer files to guarantee chronological ordering of data points,
    reversing the data if necessary. Maintains original header information
    and formatting while ensuring consistent time progression.
    
    Parameters:
    -----------
    path_to_tracers : str
        Directory path containing tracer files to be processed
        
    Returns:
    --------
    None
    """
    files = glob.glob(os.path.join(path_to_tracers, "tracer*"))

    for file_path in files:
        with open(file_path, "r") as f:
            header_lines = []
            data_lines = []
            for line in f:
                if line.startswith("#") or line.startswith("-"):
                    header_lines.append(line)
                else:
                    data_lines.append(line.strip())

        if not data_lines:
            continue

        # Determine column names from the header
        # Assume header line with column names contains the first '#' and not the 'Mass' line
        for hline in header_lines:
            if 't [' in hline:
                # Extract column names by splitting and removing units
                col_names = []
                for col in hline.strip("# \n").split():
                    name = col.split('[')[0].strip()
                    if name:  # skip empty strings
                        col_names.append(name)
                break


        # Build format list for this file
        fmt_list = [tracer_entries_fmt[name] for name in col_names]

        # Load data
        data = np.array([line.split() for line in data_lines], dtype=float)

        # Reverse if time is descending
        if data[0, 0] > data[-1, 0]:
            data = data[::-1]

        # Write back to the same file
        with open(file_path, "w") as f:
            for line in header_lines:
                f.write(line)
            for row in data:
                row_str = " ".join(fmt % val for fmt, val in zip(fmt_list, row))
                f.write(row_str + "\n")

def ensure_ascending_time_order_nse_flag(path_to_tracers, reached_NSE):
    """
    Ensures all tracer files have data ordered by ascending time and updates
    the first header line to include whether each tracer reached NSE.

    Processes tracer files to guarantee chronological ordering of data points,
    reversing the data if necessary. Maintains original header information
    and formatting while ensuring consistent time progression. Additionally,
    the first header line is updated from e.g.
        '#mass of the tracer: XYg'
    to
        '#mass of the tracer: XYg, reached_NSE: True/False'
    depending on the corresponding entry in the `reached_NSE` array.

    Parameters
    ----------
    path_to_tracers : str
        Directory path containing tracer files to be processed.
    reached_NSE : array-like of bool
        Boolean array with as many entries as tracer files. The i-th entry
        corresponds to whether tracer i reached NSE.

    Returns
    -------
    None
    """
    files = sorted(glob(os.path.join(path_to_tracers, "tracer*.dat")))

    for file_path in files:
        tracer_num = int(os.path.basename(file_path).replace("tracer", "").replace(".dat", ""))
        nse_flag = bool(reached_NSE[tracer_num])

        with open(file_path, "r") as f:
            header_lines = []
            data_lines = []
            for line in f:
                if line.startswith("#") or line.startswith("-"):
                    header_lines.append(line)
                else:
                    data_lines.append(line.strip())

        if not data_lines:
            continue

        # Update the first header line
        if header_lines:
            # Example: "#mass of the tracer: 1.23e-04g\n"
            header_lines[0] = header_lines[0].strip() + f", reached_NSE: {nse_flag}\n"

        # Determine column names from the header
        for hline in header_lines:
            if 't [' in hline:
                col_names = []
                for col in hline.strip("# \n").split():
                    name = col.split('[')[0].strip()
                    if name:
                        col_names.append(name)
                break

        # Build format list for this file
        fmt_list = [tracer_entries_fmt[name] for name in col_names]

        # Load data
        data = np.array([line.split() for line in data_lines], dtype=float)

        # Reverse if time is descending
        if data[0, 0] > data[-1, 0]:
            data = data[::-1]

        # Write back to the same file
        with open(file_path, "w") as f:
            for line in header_lines:
                f.write(line)
            for row in data:
                row_str = " ".join(fmt % val for fmt, val in zip(fmt_list, row))
                f.write(row_str + "\n")


# --------------FUNCTIONS FOR PLACING TRACERS --------------------

def PosFromFile(path_to_datfile):
    """
    Extracts position and mass data from a tracer data file.
    
    Reads a data file containing tracer information and returns
    the x-coordinates, y-coordinates, and mass values for each tracer.
    Assumes the file format has columns for x, y, and mass in order.
    
    Parameters:
    -----------
    path_to_datfile : str
        Path to the data file containing tracer positions and masses
        
    Returns:
    --------
    ptX : ndarray
        Array of x-coordinates for each tracer
    ptY : ndarray  
        Array of y-coordinates for each tracer
    ptM : ndarray
        Array of mass values for each tracer
    """
    tr_pos = np.genfromtxt(path_to_datfile)
    ptX = tr_pos[:,0]
    ptY = tr_pos[:,1]
    ptM = tr_pos[:,2]

    return ptX, ptY, ptM

def PosFromDens_Bene(snap, ptN, only_unbound=True, maxDens = None):
    """
    Place tracers proportional to mass, with optional restriction to only_unbound mass only.
    In case of only_unbound=False, cells with dens>maxDens will be excluded.
    Ensures at least one tracer per block with the given restrictions/criterion.

    Parameters
    ----------
    snap : Snapshot2D
        FLASH snapshot object.
    ptN : int
        Total number of tracers to place.
    only_unbound : bool, optional
        If True (default), only only_unbound cells are considered for placement.
        If False, all cells are considered.
    maxDens : int/float, optional
        If only_unbound=False, maxDens will be used to exclude certain regions with dens>maxDens (i.e. a PNS)
    """
    t0 = time()

    block_masses = []
    block_cell_info = []

    # --- Loop over blocks ---
    for block_id in range(len(snap.bbox)):
        # Get cell centers and edges
        x_ccenters, y_ccenters, x_clows, x_chighs, y_clows, y_chighs = snap.cellCoords(block_id, with_edges=True)
    
        # Cell properties
        cvols = snap.cellVolumes(block_id)        # shape (Ny, Nx)
        cdens = snap.td_vars['dens'][block_id]
        cvelx = snap.td_vars['velx'][block_id]
        cvely = snap.td_vars['vely'][block_id]
        cener = snap.td_vars['ener'][block_id]
        cgpot = snap.td_vars['gpot'][block_id]
    
        # Cell masses
        cmasses = cvols * cdens
    
        # Radial velocity
        r = np.sqrt(x_ccenters**2 + y_ccenters**2)
        dot = x_ccenters * cvelx + y_ccenters * cvely
        cvrad = dot / np.maximum(r, 1e-20)
    
        # Mask: only_unbound or not
        if only_unbound:
            mask = (cener + cgpot > 0) & (cvrad > 0)
        else:
            mask = (cdens < maxDens) # if not only only_unbound cells, than not in areas with dens>maxDens (i.e. not in PNS)
    
        # Flatten arrays
        cmasses_flat = cmasses.flatten()
        xlow_flat = np.tile(x_clows, snap.cells_per_block_y)
        xhigh_flat = np.tile(x_chighs, snap.cells_per_block_y)
        ylow_flat = np.repeat(y_clows, snap.cells_per_block_x)
        yhigh_flat = np.repeat(y_chighs, snap.cells_per_block_x)
        mask_flat = mask.flatten()
    
        # Apply mask
        masses = cmasses_flat[mask_flat]
        x_low = xlow_flat[mask_flat]
        x_high = xhigh_flat[mask_flat]
        y_low = ylow_flat[mask_flat]
        y_high = yhigh_flat[mask_flat]

        total_block_mass = masses.sum()

        block_masses.append(total_block_mass)
        block_cell_info.append((masses, x_low, x_high, y_low, y_high))

    M_sun_g = M_sun.value*1e3

    total_mass = sum(block_masses)
    write_log(path_to_tracers, f"Total selected mass: {np.round(total_mass / M_sun_g, 3)} M_sun")

    # --- Assign tracers to blocks ---
    tracers_per_block = []
    approx_mass_p_tr = total_mass / ptN

    for bm in block_masses:
        tr = int(round(bm / approx_mass_p_tr))
        tracers_per_block.append(max(tr, 1))  # at least one per block

    # --- Place tracers inside blocks ---
    ptX, ptY, ptM = [], [], []

    for block_id, ntr in enumerate(tracers_per_block):
        masses, x_low, x_high, y_low, y_high = block_cell_info[block_id]

        if len(masses) == 0:
            continue  # no (only_unbound) mass in this block

        # Normalize masses for probability distribution
        probs = masses / masses.sum()

        block_mass = block_masses[block_id]
        tracer_mass = block_mass/ntr
        
        # Pick cells according to mass weights
        chosen_idx = np.random.choice(len(masses), size=ntr, p=probs)

        for idx in chosen_idx:
            ptX.append(random.uniform(x_low[idx], x_high[idx]))
            ptY.append(random.uniform(y_low[idx], y_high[idx]))
            ptM.append(tracer_mass)  # or assign ~approx_mass_p_tr if you want equal weights

    write_log(path_to_tracers, f"Placed {len(ptX)} tracers in {(time()-t0):.2f}s")

    return np.array(ptX), np.array(ptY), np.array(ptM)


# -------------------- INTEGRATION FUNCTIONS --------------------

def integrate_single_tracer(args):
    """
    Integrates the trajectory of a single tracer particle through the simulation
    domain using velocity field interpolation from FLASH snapshots.

    The function wraps around `solve_ivp` to compute tracer paths, including:
      - Handling forward or backward integration in time (`sgn`).
      - Interpolating velocity fields between snapshot times and positions.
      - Terminating integration when tracers leave the simulation domain (OOB events).
      - Enforcing wall-clock timeouts to prevent runaway calculations.
      - Sampling physical quantities along the tracer path and writing them to output.
      - Detecting whether the tracer reached nuclear statistical equilibrium (NSE)
        by checking if the maximum sampled temperature exceeds `max_temp`.

    Parameters
    ----------
    args : tuple
        A packed tuple containing:
            tr_id : int
                Unique tracer ID.
            start_pos : tuple of float
                Starting position (x, y) of the tracer in simulation units.
            times_chunk : ndarray
                Sorted array of times for the current snapshot chunk.
            snapshots : list
                List of `Snapshot2D` objects for this time chunk.
            sgn : int
                Time integration direction, +1 for forward, -1 for backward.
            keys : list of str
                Quantities to sample along the tracer trajectory (e.g. 'rho', 'temp').
            tracer_entries : dict-like
                Shared structure for storing tracer results.
            solve_ivp_args : tuple
                Numerical tolerances and step control (rtol, atol, max_step).
            still_calc : multiprocessing.Manager().dict
                Shared dict tracking whether tracers are still active.
            reached_NSE : multiprocessing.Manager().list
                Shared list marking whether each tracer has reached NSE.
            still_check_NSE : multiprocessing.Manager().list
                Shared list marking whether NSE criterion should still be checked
                for each tracer.
            output_dir : str
                Path to directory for logs and tracer output files.
            tracer_entries_fmt : str
                Format specifier for tracer output.
            time_limit : float
                Wall-clock time limit (seconds) for a single tracer integration.

    Returns
    -------
    tuple
        (final_pos, local_oob_events)
        final_pos : ndarray of shape (2,)
            Final tracer position (x, y) in simulation units after integration
            or NaN if failed.
        local_oob_events : list of str
            Log messages for OOB (out-of-bound) events or errors.

    Raises: 
    ------- 
        RuntimeError If computation exceeds wall-clock time limit for tracer integration

    Notes
    -----
    - NSE detection: If `sgn == -1` (backward integration) and `still_check_NSE`
      is True for a tracer, the function checks whether the maximum sampled
      temperature `max(vars_sampled['temp'])` exceeds the threshold `max_temp`.
      If so:
        * `reached_NSE[tr_id]` is set to True.
        * `still_check_NSE[tr_id]` is set to False (no further checking).
        * If `only_until_NSE` is True, the tracer is terminated early to save
          computation time and logged accordingly.
    - Any exception during integration or sampling is caught, logged, and
      flagged in `still_calc` so that the tracer will not be processed further.
    """
    
    try:
        (tr_id, start_pos, times_chunk, snapshots, sgn, keys, tracer_entries, solve_ivp_args, 
         still_calc, reached_NSE, still_check_NSE, output_dir, tracer_entries_fmt, time_limit) = args
        
        local_oob_events = []

        tracer_start_time = time()

        def velocity_field(t, pos):
            """
            Interpolates velocity components at a given position and time through interpolation.
            
            Interpolates velocity field values from snapshot data for tracer integration,
            handling both forward and backward time integration with boundary checks
            and time-out protection. Provides linear interpolation between snapshots
            when the requested time falls between available data points.
            
            Parameters:
            -----------
            t : float
                Time at which to evaluate the velocity field
            pos : tuple
                Position coordinates (x, y) where velocity is evaluated
                
            Returns:
            --------
            list
                Velocity components [vx, vy] at the specified position and time
                
            Raises:
            -------
            RuntimeError
                If computation exceeds wall-clock time limit for tracer integration
            """
            x, y = pos
            
            if (time()-tracer_start_time) > time_limit:
                write_log(output_dir, f'        Tracer {tr_id}: Wall-clock timeout ({time_limit:.1f}s) in solver (t,x,y) = ({t:.5f}s, {x/1e5:.6e}km, {y/1e5:.6e}km)')
                raise RuntimeError(f'Tracer {tr_id}: Wall-clock timeout ({time_limit:.1f}s) in solver (t,x,y) = ({t:.5f}s, {x/1e5:.6e}km, {y/1e5:.6e}km)')

            #TODO: properly comment this
            if sgn == 1:
                if t <= times_chunk[0]:
                    idx = 0
                    if t<times_chunk[0]:
                        write_log(output_dir,f't too small, Tracer {tr_id} accesses t={t:.8f} in chunk \in {times_chunk[0]:.8f} - {times_chunk[-1]:.8f}s')
                elif t >= times_chunk[-1]:
                    idx = -1
                    if t>times_chunk[-1]:
                        write_log(output_dir,f't too big, Tracer {tr_id} accesses t={t} in chunk \in {times_chunk[0]} - {times_chunk[-1]:.8f}s')
                else:
                    idx_r = np.searchsorted(times_chunk, t, side='right')
                    idx_l = idx_r - 1
                    t_pair = [times_chunk[idx_l], times_chunk[idx_r]]
                    velx_pair = [snapshots[idx_l].getQuantAtPos('velx', x, y, tr_id),
                                snapshots[idx_r].getQuantAtPos('velx', x, y, tr_id)]
                    vely_pair = [snapshots[idx_l].getQuantAtPos('vely', x, y, tr_id),
                                snapshots[idx_r].getQuantAtPos('vely', x, y, tr_id)]
                    return [np.interp(t, t_pair, velx_pair), np.interp(t, t_pair, vely_pair)]
                
                return [snapshots[idx].getQuantAtPos('velx', x, y, tr_id),
                        snapshots[idx].getQuantAtPos('vely', x, y, tr_id)]
            else: #if sgn == -1 aka if backwards
                if t <= times_chunk[-1]:
                    idx = -1
                    if t<times_chunk[-1]:
                        write_log(output_dir,f'Tracer {tr_id} accesses t={t} in chunk \in {times_chunk[0]:.6f} - {times_chunk[-1]:.6f}s')#fmt for t : :.6f
                elif t >= times_chunk[0]:
                    idx = 0
                    if t>times_chunk[0]:
                        write_log(output_dir,f'Tracer {tr_id} accesses t={t} in chunk \in {times_chunk[0]:.6f} - {times_chunk[-1]:.6f}s')
                else:
                    times_chunk_asc = times_chunk[::-1] #searchsorted only works with ascendingly ordered arrays
                    idx_asc = np.searchsorted(times_chunk_asc, t, side='right')
                    idx_r = len(times_chunk)-idx_asc
                    idx_l = idx_r - 1 #did the check, indices righly chosen
                    t_pair = [times_chunk[idx_l], times_chunk[idx_r]]
                    
                    velx_pair = [snapshots[idx_l].getQuantAtPos('velx', x, y, tr_id),
                                snapshots[idx_r].getQuantAtPos('velx', x, y, tr_id)]
                    vely_pair = [snapshots[idx_l].getQuantAtPos('vely', x, y, tr_id),
                                snapshots[idx_r].getQuantAtPos('vely', x, y, tr_id)]
                    
                    interfunc_x = interp1d(t_pair, velx_pair)
                    interfunc_y = interp1d(t_pair, vely_pair)

                    return [interfunc_x(t), interfunc_y(t)] #Double checked, is right
                
                return [snapshots[idx].getQuantAtPos('velx', x, y, tr_id),
                        snapshots[idx].getQuantAtPos('vely', x, y, tr_id)]


        def oob_event(t, pos):
            x,y = pos
            return min(x, xmax-x, y-ymin, ymax-y) #-eps + 1e-10 try out iff not converging       #if tracer leaves simulation domain this is <0 and gets handled from solve_ivp

        oob_event.terminal = True                       #solve_ivp should terminate calculation if triggered
        oob_event.direction = -1                        #triggers when tracer leaves the sim domain


    
        def sample_vars_nearest(times, positions, keys, snapshots, times_chunk):
            n = len(times)
            sampled = {key: np.empty(n) for key in keys}
            for i, (t, x, y) in enumerate(zip(times, positions[0], positions[1])):
                idx = np.abs(times_chunk - t).argmin()
                snap = snapshots[idx]
                for key in keys:
                    sampled[key][i] = snap.getQuantAtPos(key, x, y, tr_id)
            
            return sampled
        t_span = [times_chunk[0], times_chunk[-1]]


        t_eval = times_chunk
        start_pos= [start_pos[0], start_pos[1]]

        #main part: the integration
        result = solve_ivp(velocity_field,
                            t_span=t_span,
                            y0=start_pos, #TODO:t_eval=t_eval rauslassen testen, t_eval=t_eval,
                            rtol=solve_ivp_args[0],
                            t_eval=t_eval,
                            atol=solve_ivp_args[1],
                            max_step=solve_ivp_args[2],
                            method='RK45',
                            first_step=1e-6,
                            events=oob_event)



        #handle oob_events:
        if result.t_events[0].size > 0:
            still_calc[tr_id] = False
            exit_time = result.t_events[0][0]
            exit_pos = result.y_events[0][0]

            # Logging
            if exit_pos is not None:
                log_pos = f"{np.round(exit_pos[0]/1e5,2)}, {np.round((exit_pos[1])/1e5,2)}"
            else:
                log_pos = "unknown"
            write_log(output_dir, f"        Tracer {tr_id} OOB at t={exit_time:.3f}, (x,y)=({log_pos}) km")
            local_oob_events.append(f'      Tracer {tr_id} at t={np.round(exit_time, 3)}, (x,y)=({log_pos}) km')

            t_vals = result.t
            x_vals = result.y[0]
            y_vals = result.y[1]

            # Sample variables for in-bounds time            
            try:
                vars_sampled = sample_vars_nearest(t_vals,
                                                    (x_vals, y_vals),
                                                    keys,
                                                    snapshots,
                                                    times_chunk)
                #if direction = backwards and only_until_NSE and temp > NSE_temp: mark in reached_NSE and terminate tracer to save computation time
                if sgn == -1 and  still_check_NSE and 'temp' in vars_sampled:
                    if max(vars_sampled['temp']) > max_temp:
                        reached_NSE[tr_id] = True
                        still_check_NSE[tr_id] = False
                        if only_until_NSE:
                            still_calc[tr_id] = False
                            write_log(output_dir, f'        Tracer {tr_id} reached max_temp = {max_temp/1e9:.1f} GK and will be terminated')

            except Exception as inner_e:
                write_log(output_dir, f"‼️ sample_vars_nearest failed for tracer {tr_id}: {inner_e}")
                raise

            write_to_tracer_file(t_vals, [x_vals, y_vals], keys, vars_sampled, tracer_entries, tr_id, output_dir, tracer_entries_fmt)

            return np.array([x_vals[-1], y_vals[-1]]), local_oob_events

        
        vars_sampled = sample_vars_nearest(result.t, result.y, keys, snapshots, times_chunk)

        #if direction is backwards and only_until_NSE and temp > NSE_temp: mark in reached_NSE and terminate tracer to save computation time
        #TODO: function check_NSE that handles this - until now reduntant in events block and here
        if sgn == -1 and still_check_NSE and 'temp' in vars_sampled:
            if max(vars_sampled['temp']) > max_temp:
                reached_NSE[tr_id] = True
                still_check_NSE[tr_id] = False
                if only_until_NSE:
                    still_calc[tr_id] = False
                    write_log(output_dir, f'        Tracer {tr_id} reached max_temp = {max_temp/1e9:.1f} GK and will be terminated')
        write_to_tracer_file(result.t, result.y, keys, vars_sampled, tracer_entries, tr_id, output_dir, tracer_entries_fmt)

        return np.array([result.y[0][-1], result.y[1][-1]]), local_oob_events
    
    except Exception as e:
        tb_str = traceback.format_exc()
        write_log(output_dir,f"❌ Tracer {args[0]} failed: {e}, tb_str: {tb_str}")
        still_calc[args[0]] = False  # Mark as failed and dont calc further
        return np.array([np.nan, np.nan]), [f"❌ Tracer {args[0]} failed: {e}"]

def integrate_chunk(chunk_args):

    (plt_files_chunk, start_pos, tracer_entries, solve_ivp_args, still_calc, reached_NSE, still_check_NSE, output_dir,
        tracer_entries_fmt, single_tracer_tlim) = chunk_args

    t1 = time()
    snapshots = [Snap.Snapshot2D(f, keys) for f in plt_files_chunk]
    times_chunk = [snap.currentSimTime() for snap in snapshots]

    _, unique_indices = np.unique(times_chunk, return_index=True)
    snapshots = [snapshots[i] for i in sorted(unique_indices)]
    times_chunk = np.array([times_chunk[i] for i in sorted(unique_indices)])

    t2 = time()
    write_log(output_dir,f'Read {len(snapshots)} snapshots in {np.round(t2 - t1, 2)} s')

    write_log(output_dir,
        f"Chunk {start} → {end}: Times = [{times_chunk[0]:.3f} - {times_chunk[-1]:.3f}] "
        f"({len(times_chunk)} steps)"
    )

    active_tracer_ids = np.where(still_calc)[0]

    task_args = [
        (tr_id, start_pos[tr_id], times_chunk, snapshots, sgn, keys, tracer_entries, solve_ivp_args,
         still_calc, reached_NSE, still_check_NSE, output_dir, tracer_entries_fmt, single_tracer_tlim)
        for tr_id in active_tracer_ids
    ]

    # Create and destroy pool per chunk
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=num_cpus)

    try:
        results = pool.map(integrate_single_tracer, task_args)
        partial_new_positions = [res[0] for res in results]
        partial_oob_events = [res[1] for res in results]
    finally:
        pool.close()
        pool.join()  # Wait for all worker processes to exit cleanly

    new_positions = np.full((len(start_pos), 2), np.nan) #initialize array full of nan's
    
    for idx, tr_id in enumerate(active_tracer_ids): #still active tracers override their nan's to the new position, necessary to keep tr_id's between chunks
        new_positions[tr_id] = partial_new_positions[idx]

    t3 = time()
    write_log(output_dir,f'Integrated chunk in {np.round(t3 - t2, 2)} s')
    
    for events in partial_oob_events:
        oob_list.extend(events)
 

    return np.array(new_positions)

# -------------------- MAIN EXECUTION --------------------

if __name__ == "__main__":

    t0 = time()

    mp_manager = mp.Manager()

    #mp.set_start_method('spawn', force=True) 

    if not os.path.exists(path_to_tracers):
        os.makedirs(path_to_tracers)

    if len(os.listdir(path_to_tracers)) > 1:
        print(f"Directory already exists and is not empty: {path_to_tracers}. Exiting")
        sys.exit(1)

    #Write chosen parameters into the run.log file
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) #to now which slurm out for which run
    write_log(path_to_tracers, f"\n")
    write_log(path_to_tracers, f"Starting tracer calculation...")
    write_log(path_to_tracers, f"Chosen Parameters: ")
    write_log(path_to_tracers, f"       Integration direction = {direction}")
    write_log(path_to_tracers, f"       calc_seeds = {calc_seeds}")
    write_log(path_to_tracers, f"       with_neutrinos = {with_neutrinos}")
    write_log(path_to_tracers, f"       Tracer placement = {placement_method}")

    if placement_method == 'PosWithDens':
        write_log(path_to_tracers, f"       num_tracers = {num_tracers}")
        write_log(path_to_tracers, f"       only_unbound = {only_unbound}")
        if only_unbound==False:
            write_log(path_to_tracers, f"       max_dens = {max_dens:.1e}")
    
    write_log(path_to_tracers, f'       only_until_NSE = {only_until_NSE}')
    if only_until_NSE:
        write_log(path_to_tracers, f'       max_temp = {max_temp/1e9:.2e} GK')

    write_log(path_to_tracers,f"       used tolerances: rtol= {rtol:.1e}, atol={atol:.1e}, maxstep={maxstep:.1e}")
    write_log(path_to_tracers, f"\n")

    if arb_message: #check to see its not empty e.g. ""
        write_log(path_to_tracers, 'User message: '+ arb_message)

    #----------------CHOOSE KEYS FOR SNAPSHOTS --------------

    if with_neutrinos:
        keys = ['dens', 'temp', 'ye', 'entr', 'velx', 'vely', 'gpot', 'ener', 'fnue', 'fnua', 'fnux', 'enue', 'enua', 'enux']
        tracer_entries = ['r', 'dens', 'temp', 'ye', 'entr', 'velx', 'vely', 'lnue', 'lnua', 'lnux', 'lanux', 'enue', 'enua', 'enux', 'eanux', 'ejected']
    else:
        keys = ['dens', 'temp', 'ye', 'entr', 'velx', 'vely', 'gpot', 'ener']
        tracer_entries = ['r', 'dens', 'temp', 'ye', 'entr', 'velx', 'vely', 'ejected']


    tracer_entries_units = {'t': 's',
                            'x': 'cm',
                            'y': 'cm',
                            'r': 'km',
                            'dens': 'g/cm^3',
                            'temp': 'GK',
                            'entr': 'k_B',
                            'ye' : '',
                            'velx': 'cm/s',
                            'vely': 'cm/s',
                            'lnue': 'erg/s',
                            'lnua': 'erg/s',
                            'lnux': 'erg/s',
                            'lanux': 'erg/s',
                            'enue': 'MeV',
                            'enua': 'MeV',
                            'enux': 'MeV',
                            'eanux': 'MeV',
                            }
    
    tracer_entries_fmt={'t': '%.4f',
                        'x': '%.6e',
                        'y': '%.6e',
                        'r': '%.6e', 

                        'dens': '%.6e',
                        'ye' : '%.4f',
                        'temp': '%.6e',
                        'entr': '%.4f',
                        'velx': '%.6e',
                        'vely': '%.6e',
                        'lnue': '%.6e',
                        'lnua': '%.6e',
                        'lnux': '%.6e',
                        'lanux': '%.6e',
                        'enue': '%.6e',
                        'enua': '%.6e',
                        'enux': '%.6e',
                        'eanux': '%.6e',
                        'ejected': '%d'
                        }

    #----------------- GET RESOURCES FROM SLURM -------------
    try:
        num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
        if num_cpus is None:
            raise ValueError
    except (TypeError, ValueError):
        # Fallback if not running under Slurm or variable missing
        num_cpus = 1

    write_log(path_to_tracers, f"Using {num_cpus} CPUs")

    if direction == 'backward':
        sgn = -1
    else:
        sgn = 1

    # Initialize snapshot and place tracers according to chosen integration direction and placement method
    if direction == 'backward':
        intr_start = Snap.Snapshot2D(plt_files[sgn], keys=keys)
        if placement_method == 'PosWithDens':
            init_x, init_y, init_mass = PosFromDens_Bene(intr_start, num_tracers, only_unbound=only_unbound, maxDens=max_dens)
        elif placement_method == 'FromFile':
            init_x, init_y, init_mass = PosFromFile(path_to_tracer_start)
        else:
            write_log(path_to_tracers, 'Unknown tracer placement method. Exiting.')
            sys.exit('Unknown tracer placement method. Exiting.')   
    elif direction == 'forward':
        intr_start = Snap.Snapshot2D(plt_files[0], keys=keys)
        if placement_method == 'PosWithDens':
            init_x, init_y, init_mass = PosFromDens_Bene(intr_start, num_tracers, only_unbound=only_unbound, maxDens=max_dens)
        elif placement_method == 'FromFile':
            init_x, init_y, init_mass = PosFromFile(path_to_tracer_start)
        else:
            write_log(path_to_tracers, 'Unknown tracer placement method. Exiting.')
            sys.exit('Unknown tracer placement method. Exiting.')
    else:
        write_log(path_to_tracers, 'Unknown integration direction. Exiting.')
        sys.exit('Unknown integration direction. Exiting.')

    initial_pos = np.column_stack((init_x, init_y))

    placed_num_tracers = len(init_mass) #override num_tracers to get real number of placed tracers

    still_calc = mp_manager.Array('b', [True] * placed_num_tracers)  # Shared across all processes
    reached_NSE = mp_manager.Array('b', [False] * placed_num_tracers)
    still_check_NSE = mp_manager.Array('b', [True] * placed_num_tracers)
    # ----------OPEN TRACER FILES AND WRITE HEADER ----------------------

    t1 = time()

    # --- build header ---
    header_cols = []

    # function to get width from fmt
    def fmt_width(fmt):
        if fmt.endswith('e'):
            return 12
        elif fmt.endswith('f'):
            return 5
        elif fmt.endswith('d'):
            return 1
        else:
            return len(fmt)

    # t, x, y first
    for key in ['t','x','y']:
        width = fmt_width(tracer_entries_fmt[key])
        unit = tracer_entries_units.get(key, '')
        col_str = f"{key} [{unit}]"
        padding = width + 1 - len(col_str)
        if padding < 1: padding = 1
        header_cols.append(col_str + ' '*padding)

    # remaining tracer entries
    for key in tracer_entries:
        fmt = tracer_entries_fmt.get(key, '%.6e')
        width = fmt_width(fmt)
        unit = tracer_entries_units.get(key, '')
        col_str = f"{key} [{unit}]"
        padding = width + 1 - len(col_str)
        if padding < 1: padding = 1
        header_cols.append(col_str + ' '*padding)

    header = "# " + "".join(header_cols)
    separator = "# " + '-' * (len(header))

    # --- loop over tracers and write files ---
    for tr_id in range(placed_num_tracers):
        filename = os.path.join(path_to_tracers, f"tracer{str(tr_id).zfill(5)}.dat")
        with open(filename, "w") as f:
            f.write(f"# Mass of the tracer [g]: {init_mass[tr_id]:.6e}\n")
            f.write(header + "\n")
            f.write(separator + "\n")

    t2 = time()
    write_log(path_to_tracers, f"Tracer files created in {np.round(t2-t1, 2)}s ")


    # ------------------ EXECUTING MAIN LOOP --------------------------

    oob_list = []

    n_chunks = len(plt_files) // chunk_size
    write_log(path_to_tracers,f"Integrating {direction} in {n_chunks} chunks:")
    write_log(path_to_tracers,"____________________________________________________ \n")

    if sgn == 1:
        chunk_ranges = range(0, len(plt_files), chunk_size)
    else:
        chunk_ranges = range(len(plt_files) -1 , -1, -chunk_size)

    pos = initial_pos

    for start in chunk_ranges:
        if sgn == 1:
            end = min(start + chunk_size, len(plt_files))
            plt_chunk = plt_files[start:end]
        else:
            end = max(start - chunk_size + 1, 0)
            plt_chunk = plt_files[end:start + 1][::-1]

        write_log(path_to_tracers,f'Starting chunk from index {start} to {end}')
        active_before = sum(still_calc)
        chunk_args =   (plt_chunk, pos,  tracer_entries, solve_ivp_args, still_calc, reached_NSE,
                        still_check_NSE, path_to_tracers, tracer_entries_fmt, PER_TRACER_TIMEOUT)
        pos = integrate_chunk(chunk_args=chunk_args)
        active_after = sum(still_calc)
        write_log(path_to_tracers,f"Active tracers after chunk: {active_after}\n")

    write_log(path_to_tracers,f'{len(oob_list)} OOB events:')

    for oob_event in oob_list:
        write_log(path_to_tracers,f'  {oob_event}')

    write_log(path_to_tracers,f"Integration complete. Total timesteps: {len(plt_files)}")

    if direction == 'backward':
        write_log(path_to_tracers, f'Assuring ascending time order for nuclear network')
        ensure_ascending_time_order_nse_flag(path_to_tracers, reached_NSE)

    if calc_seeds:
        write_log(path_to_tracers, f'Starting to calculate initial compositions of tracers')
        seeds_dir = os.path.join(path_to_tracers, 'seeds')
        if not os.path.exists(seeds_dir):
            os.makedirs(seeds_dir)  # create the seeds directory
        
        tracers = sorted(glob(os.path.join(path_to_tracers, "tracer*")))

        if prog_type == 'NuGrid':
            progenitor = Prog.Progenitor_NuGrid(path_to_progfile=path_to_progfile)
        elif prog_type == 'FLASH':
            progenitor = Prog.Progenitor_FLASH(path_to_progfile=path_to_progfile)

        for tr_id, tracer in enumerate(tracers):
            tr_dat = np.genfromtxt(tracer, skip_header=3, usecols=[0, 1, 2])
            t_min_idx = np.argmin(tr_dat[:,0])
            tr_init_radius = np.sqrt(tr_dat[t_min_idx, 1]**2 + tr_dat[t_min_idx, 2]**2)
            
            massfrac_dict = progenitor.massfractions_of_r(tr_init_radius)

            data = np.array([[info['A'], info['Z'], info['X']] for info in massfrac_dict.values()])
            data = data[data[:,0].argsort()]

            filename = os.path.join(path_to_tracers, 'seeds', f'seed{str(tr_id).zfill(5)}.txt')
            np.savetxt(filename, data, fmt="%d\t%d\t%.6e", header="# A\tZ\tX\n# ---------------", comments='')
        
        write_log(path_to_tracers, f'Calculated and saved the initial compositions of the tracers')

    write_log(path_to_tracers, f'Tracer calculation complete :) - total time {(time()-t0)/60:.2f} minutes.')
