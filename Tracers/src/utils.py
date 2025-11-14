# ==============================================================================
#  2-D FLASH TRACER INTEGRATION
#  Author: Benedikt Weinhold  (~Oct 2025)
#
#  UTILS
#
#  Various functions to help in file writing, shared memory setup, ...
#  For further explanations see README.md 
# ==============================================================================

import os, psutil
from datetime import datetime
import numpy as np
from multiprocessing.shared_memory import SharedMemory
import time

import src.Snapshot2D as Snap
import src.Progenitors as Prog

from config import *

# ----------------------- LOGFILE WRITING --------------------------------------


def write_log(output_dir, msg):
    """
    Append a message to the run log file with a timestamp.

    Creates the log file if it doesn't exist and ensures the output directory exists.

    Parameters
    ----------
    output_dir : str
        Directory where 'run.log' will be created or appended.
    msg : str
        Message to write to the log.
    """
    logfile_path = os.path.join(output_dir, 'run.log')
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(logfile_path, 'a') as f:
        f.write(f'[{timestamp}] {msg}\n')
        f.flush()

def create_progress_bar(completed, total, bar_length=20):
    """
    Generate a text-based progress bar for displaying task completion.

    Parameters
    ----------
    completed : int
        Number of completed tasks.
    total : int
        Total number of tasks.
    bar_length : int, optional
        Length of the progress bar in characters (default: 20).

    Returns
    -------
    str
        Text progress bar with percentage and counts.
    """
    percent = (completed / total) * 100
    filled = int(bar_length * completed / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    return f"[{bar}] {completed}/{total} ({percent:.1f}%)"

def log_used_params():
    """Writes all chosen params in the run.log file"""
    write_log(PATH_TO_OUTPUT, f"\n")
    write_log(PATH_TO_OUTPUT, f"Starting tracer calculation...")
    write_log(PATH_TO_OUTPUT, f"Chosen Parameters: ")
    write_log(PATH_TO_OUTPUT, f"       Integration direction = {DIRECTION}")
    write_log(PATH_TO_OUTPUT, f"       calc_seeds = {CALC_SEEDS}")
    write_log(PATH_TO_OUTPUT, f"       with_neutrinos = {WITH_NEUTRINOS}")
    write_log(PATH_TO_OUTPUT, f"       Tracer placement = {PLACEMENT_METHOD}")

    if PLACEMENT_METHOD == 'PosWithDens':
        write_log(PATH_TO_OUTPUT, f"       num_tracers = {NUM_TRACERS}")
        write_log(PATH_TO_OUTPUT, f"       max_temp_place = {MAX_TEMP_PLACE/1e9:.1e} GK")
        write_log(PATH_TO_OUTPUT, f"       ye_steps = {YE_STEPS}")
        write_log(PATH_TO_OUTPUT, f"       only_unbound = {ONLY_UNBOUND}")
        if ONLY_UNBOUND==False:
            write_log(PATH_TO_OUTPUT, f"       max_dens = {MAX_DENS_PLACE:.1e} g/cm³")
    if PLACEMENT_METHOD == 'FromFile':
        write_log(PATH_TO_OUTPUT, f"       path to start positions: {PATH_TO_TRACERS_START}")
    
    write_log(PATH_TO_OUTPUT, f'       only_until_maxTemp = {ONLY_UNTIL_MAXTEMP}')
    if ONLY_UNTIL_MAXTEMP:
        write_log(PATH_TO_OUTPUT, f'       max_temp = {MAXTEMP_TRACER/1e9:.1e} GK')

    write_log(PATH_TO_OUTPUT,f"       used tolerances: rtol= {RTOL:.1e}, atol={ATOL:.1e}, maxstep={MAXSTEP:.1e}")
    write_log(PATH_TO_OUTPUT, f"\n")

    if ARB_MESSAGE: #check to see its not empty e.g. ""
        write_log(PATH_TO_OUTPUT, 'User message: '+ ARB_MESSAGE) 

# ----------------------- PHYSICAL CRITERIA ------------------------------------

def is_ejected(gpot, ener, velx, vely, x, y):
    """
    Mass counts as ejected IFF:
        i)  gravitationally unbound (ener+gpot > 0) and
        ii) outwards moving (v_radial > 0)
    """
    r = np.sqrt(x**2 + y**2)
    r_safe = np.maximum(r, 1e-30)
    v_r = (velx * x + vely * y) / r_safe
    return ((v_r > 0) & ((ener + gpot) > 0)).astype(int)

def L_from_f(f, r):
    """
    Calculates local neutrino luminosity from flux f and radius r
    f [erg/s cm^2], r [cm] -> L [erg/s] or 
    f [B/s cm^2], r [cm] -> L [B/s]
    """
    return 4 * np.pi * r**2 * f

# ------------------- CALCULATE INITIAL COMPOSITION ------------------------

def calc_seeds():
    """
    Calculate and save initial compositions for all tracers.

    For each tracer, determines the initial radius, retrieves the corresponding
    mass fractions from the progenitor model, and saves them in a seeds/ directory.
    Logs progress at the start and end of the calculation.

    Notes
    -----
    Supports NuGrid or FLASH progenitor types based on the PROG_TYPE setting.
    Each seed file is named seed#####.txt and contains columns A, Z, X.
    """

    seeds_dir = os.path.join(PATH_TO_OUTPUT, 'seeds')
    if not os.path.exists(seeds_dir):
        os.makedirs(seeds_dir)  # create the seeds directory
    
    tracers = sorted(glob(os.path.join(PATH_TO_OUTPUT, "tracers/tracer*")))

    if PROG_TYPE == 'NuGrid':
        progenitor = Prog.Progenitor_NuGrid(path_to_progfile=PATH_TO_PROGFILE)
    elif PROG_TYPE == 'FLASH':
        progenitor = Prog.Progenitor_FLASH(path_to_progfile=PATH_TO_PROGFILE)

    for tr_id, tracer in enumerate(tracers):
        tr_dat = np.genfromtxt(tracer, skip_header=3, usecols=[0, 1, 2])
        t_min_idx = np.argmin(tr_dat[:,0])
        tr_init_radius = np.sqrt(tr_dat[t_min_idx, 1]**2 + tr_dat[t_min_idx, 2]**2)
        
        massfrac_dict = progenitor.massfractions_of_r(tr_init_radius)

        data = np.array([[info['A'], info['Z'], info['X']] for info in massfrac_dict.values()])
        data = data[data[:,0].argsort()]

        filename = os.path.join(PATH_TO_OUTPUT, 'seeds', f'seed{str(tr_id).zfill(5)}.txt')
        np.savetxt(filename, data, fmt="%d\t%d\t%.6e", header="# A\tZ\tX\n# ---------------", comments='')
    
    write_log(PATH_TO_OUTPUT, f'Calculated and saved the initial compositions of the tracers')


# ----------------------- SHARED MEMORY ------------------------------------

# def load_snapshots_into_shm(file_list, td_vars_keys, sgn):
#     meta_list = []
#     time_list = []
#     shm_list = []

#     write_log(PATH_TO_OUTPUT, "Reading in snapshots into shared memory...")
#     t0 = time.time()

#     prev_time = None

#     for plt_path in file_list:

#         snap = Snap.Snapshot2D(plt_path, td_vars_keys)
#         snap_time = snap.currentSimTime()

#         #check monotonicity of snapshots
#         if prev_time is not None:
#             if sgn * (snap_time - prev_time) <= 0:
#                 print(f'Snapshot {plt_path} breaks time monotonicity of snapshots, skipping it')
#                 continue
        
#         prev_time = snap_time
#         time_list.append(snap.currentSimTime())

#         # Automatically create SHM for all attributes
#         snap_meta, snap_shms = create_shm_for_snapshot_generic(snap, td_vars_keys)
#         meta_list.append(snap_meta)
#         shm_list.extend(snap_shms)

#     write_log(PATH_TO_OUTPUT, f'Chunk-times: {time_list[0]:.3f} - {time_list[-1]:.3f}s')
#     write_log(PATH_TO_OUTPUT, f"Snapshots loaded in {time.time() - t0:.2f}s")
#     process = psutil.Process(os.getpid())
#     write_log(PATH_TO_OUTPUT, f"Current memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
#     return meta_list, time_list, shm_list

def load_snapshots_into_shm(file_list, td_vars_keys, sgn):
    """
    Load FLASH snapshots into shared memory for parallel processing.

    Reads a list of snapshot files, ensures they are monotonically increasing
    in simulation time, creates shared memory arrays for the requested variables,
    and logs progress and memory usage.

    Parameters
    ----------
    file_list : list of str
        Paths to snapshot files to load.
    td_vars_keys : list of str
        Names of the simulation variables to store in shared memory.
    sgn : int
        Sign for time monotonicity check (+1 for increasing time, -1 for decreasing).

    Returns
    -------
    meta_list : list
        Metadata for each snapshot.
    time_list : list of float
        Simulation times corresponding to each snapshot.
    shm_list : list
        Shared memory objects for all requested variables across snapshots.
    """

    meta_list = []
    time_list = []
    shm_list = []
    
    write_log(PATH_TO_OUTPUT, "Reading in snapshots into shared memory...")
    
    start_time = time.time()
    last_log = start_time
    log_interval_sec = LOG_EVERY
    
    n_files = len(file_list)
    prev_time = None
    
    for i, plt_path in enumerate(file_list, start=1):
        snap = Snap.Snapshot2D(plt_path, td_vars_keys)
        snap_time = snap.currentSimTime()
        
        # Check monotonicity of snapshots
        if prev_time is not None:
            if sgn * (snap_time - prev_time) <= 0:
                print(f'Snapshot {plt_path} breaks time monotonicity of snapshots, skipping it')
                continue
        prev_time = snap_time
        
        time_list.append(snap.currentSimTime())
        
        # Automatically create SHM for all attributes
        snap_meta, snap_shms = create_shm_for_snapshot_generic(snap, td_vars_keys)
        meta_list.append(snap_meta)
        shm_list.extend(snap_shms)
        
        # --- Time-based progress logging ---
        now = time.time()
        if now - last_log >= log_interval_sec or i == n_files:
            elapsed = now - start_time
            progress_bar = create_progress_bar(i, n_files)
            write_log(
                PATH_TO_OUTPUT,
                f"     Snapshot loading progress: {progress_bar}"
            )
            last_log = now
    
    write_log(PATH_TO_OUTPUT, f'Chunk-times: {time_list[0]:.3f} - {time_list[-1]:.3f}s')
    write_log(PATH_TO_OUTPUT, f"Snapshots loaded in {time.time() - start_time:.2f}s")
    
    process = psutil.Process(os.getpid())
    write_log(PATH_TO_OUTPUT, f"Current memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
    
    return meta_list, time_list, shm_list


def reconstruct_snapshots(meta_list):
    """Rebuild Snapshot2D objects from shared memory metadata."""
    return [Snap.Snapshot2D.from_shm(snap_meta) for snap_meta in meta_list]

def create_shm_for_array(arr):
    """
    Create a shared memory array from a NumPy array.

    Allocates shared memory for the input array so that multiple processes
    can access it without duplicating memory. Copies the data into the shared
    memory array once and returns metadata for later reference.

    Parameters
    ----------
    arr : np.ndarray
        The NumPy array to copy into shared memory.

    Returns
    -------
    meta : dict
        Metadata about the shared memory array, including name, shape, and dtype.
    shm : SharedMemory
        The actual shared memory object, which must be closed/unlinked when done.
    """
    shm = SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]  # copy data once
    meta = {
        "name": shm.name,
        "shape": arr.shape,
        "dtype": str(arr.dtype)
    }
    return meta, shm

def create_shm_for_snapshot_generic(snapshot, td_vars_keys=None):
    """
    Create shared memory arrays for all relevant data in a Snapshot2D.

    Copies all NumPy array attributes of the snapshot into shared memory so
    multiple processes can access them without duplicating memory. Optionally
    handles selected `td_vars` separately.

    Parameters
    ----------
    snapshot : Snapshot2D
        The snapshot object containing simulation data arrays and attributes.
    td_vars_keys : list of str, optional
        List of keys in snapshot.td_vars to store in shared memory.

    Returns
    -------
    shm_info : dict
        Metadata for all attributes stored in shared memory. For td_vars, returns
        a nested dict under "td_vars".
    shm_handles : list of SharedMemory
        List of shared memory objects to keep alive while in use.
    """
    td_vars_keys = td_vars_keys or []
    shm_info = {}
    shm_handles = []

    for attr_name, attr_value in snapshot.__dict__.items():
        # Skip td_vars, we'll handle them separately
        if attr_name == "td_vars":
            continue

        # Only create shared memory for numpy arrays
        if isinstance(attr_value, np.ndarray):
            meta, shm = create_shm_for_array(attr_value)
            shm_info[attr_name] = meta
            shm_handles.append(shm)
        else:
            # For scalars or other types, just store in metadata
            shm_info[attr_name] = {"value": attr_value}

    # Handle td_vars separately as before
    td_shm_info, td_shm_handles = {}, []
    if hasattr(snapshot, "td_vars") and td_vars_keys:
        for key in td_vars_keys:
            if key in snapshot.td_vars:
                arr = snapshot.td_vars[key]
                meta, shm = create_shm_for_array(arr)
                td_shm_info[key] = meta
                td_shm_handles.append(shm)
    if td_shm_info:
        shm_info["td_vars"] = td_shm_info
        shm_handles.extend(td_shm_handles)

    return shm_info, shm_handles


def cleanup_shared_memory(shm_list):
    """
    Close and remove all shared memory segments safely.

    Loops over a list of shared memory objects, closing and unlinking them.
    Handles missing or already-deleted segments gracefully and reports other issues.

    Parameters
    ----------
    shm_list : list of SharedMemory
        List of shared memory objects to clean up.
    """
    for shm in shm_list:
        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ Issue cleaning up shared memory {shm.name}: {e}")

