# ==============================================================================
#  2-D FLASH TRACER INTEGRATION
#  Author: Benedikt Weinhold  (~Oct 2025)
#
#  UTILS
#
#  Various functions to help in file writing, shared memory setup, ...
#  For further explanations see README.md 
# ==============================================================================

import os 
from datetime import datetime
import numpy as np
from multiprocessing.shared_memory import SharedMemory

# ----------------------- LOGFILE WRITING --------------------------------------

def write_log(output_dir, msg):
    logfile_path = os.path.join(output_dir, 'run.log')
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(logfile_path, 'a') as f:
        f.write(f'[{timestamp}] {msg}\n')
        f.flush()


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
    f [erg/s cm^2], r [cm] -> L [erg/s]
    """
    return 4 * np.pi * r**2 * f


# ----------------------- SHARED MEMORY ------------------------------------

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

