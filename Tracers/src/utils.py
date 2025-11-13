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

from config import *

# ----------------------- LOGFILE WRITING --------------------------------------


def write_log(output_dir, msg):
    logfile_path = os.path.join(output_dir, 'run.log')
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(logfile_path, 'a') as f:
        f.write(f'[{timestamp}] {msg}\n')
        f.flush()

def create_progress_bar(completed, total, bar_length=40):
    """Create a text-based progress bar"""
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
            write_log(PATH_TO_OUTPUT, f"       max_dens = {MAX_DENS_PLACE:.1e}")
    
    write_log(PATH_TO_OUTPUT, f'       only_until_maxTemp = {ONLY_UNTIL_MAXTEMP:.1e}')
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


# ----------------------- SHARED MEMORY ------------------------------------

def load_snapshots_into_shm(file_list, td_vars_keys, sgn):
    meta_list = []
    time_list = []
    shm_list = []

    write_log(PATH_TO_OUTPUT, "Reading in snapshots into shared memory...")
    t0 = time.time()

    prev_time = None

    for plt_path in file_list:

        snap = Snap.Snapshot2D(plt_path, td_vars_keys)
        snap_time = snap.currentSimTime()

        #check monotonicity of snapshots
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

    write_log(PATH_TO_OUTPUT, f'Chunk-times: {time_list[0]:.3f} - {time_list[-1]:.3f}s')
    write_log(PATH_TO_OUTPUT, f"Snapshots loaded in {time.time() - t0:.2f}s")
    process = psutil.Process(os.getpid())
    write_log(PATH_TO_OUTPUT, f"Current memory usage: {process.memory_info().rss / 1024**3:.2f} GB")
    return meta_list, time_list, shm_list


def reconstruct_snapshots(meta_list):
    """Rebuild Snapshot2D objects from shared memory metadata."""
    return [Snap.Snapshot2D.from_shm(snap_meta) for snap_meta in meta_list]

def create_shm_for_array(name_prefix, arr):
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
    Automatically create shared memory for all array attributes of a Snapshot2D.
    td_vars are handled separately as before.
    Returns:
        shm_info: dict mapping attribute names to metadata
        shm_handles: list of SharedMemory objects to keep alive
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
            meta, shm = create_shm_for_array(f"{attr_name}_", attr_value)
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
                meta, shm = create_shm_for_array(f"{key}_", arr)
                td_shm_info[key] = meta
                td_shm_handles.append(shm)
    if td_shm_info:
        shm_info["td_vars"] = td_shm_info
        shm_handles.extend(td_shm_handles)

    return shm_info, shm_handles

def create_shm_for_snapshot(snapshot, keys):
    shm_info = {}
    shm_handles = []
    for key in keys:
        if hasattr(snapshot, 'td_vars') and key in snapshot.td_vars:
            arr = snapshot.td_vars[key]
        elif hasattr(snapshot, key):
            arr = getattr(snapshot, key)
        else:
            raise AttributeError(f"Snapshot2D has no attribute or td_var '{key}'")

        meta, shm = create_shm_for_array(f"{key}_", arr)
        shm_info[key] = meta
        shm_handles.append(shm)

    return shm_info, shm_handles

def cleanup_shm(shm_registry):
    for snap_key, shm_info in shm_registry.items():
        for key, meta in shm_info.items():
            try:
                SharedMemory(name=meta["name"]).unlink()
            except FileNotFoundError:
                pass

def cleanup_shared_memory(shm_list):
    """Closes and unlinks all shared memory segments safely."""
    for shm in shm_list:
        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ Issue cleaning up shared memory {shm.name}: {e}")

