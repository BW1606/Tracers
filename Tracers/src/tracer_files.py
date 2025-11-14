# ==============================================================================
#  2-D FLASH TRACER INTEGRATION
#  Author: Benedikt Weinhold  (~Oct 2025)
#
#  TRACER FILES
#
#  Functions to help in tracer file writing, setting up header formats, ..
#  For further explanations see README.md 
# ==============================================================================

import os
import multiprocessing as mp
from glob import glob
import numpy as np

from .utils import is_ejected, L_from_f
from config import WITH_NEUTRINOS, PATH_TO_OUTPUT




# --------- SPECIFY FORMATS AND VARS TO READ IN FROM SNAPSHOTS -----------------

#keys are entries in the snapshot files, tracer_entries are what we want in the tracer file
keys_wNu = ['dens', 'temp', 'ye', 'entr', 'velx', 'vely', 'gpot', 'ener', 'fnue', 'fnua', 'fnux', 'enue', 'enua', 'enux']
tracer_entries_wNu = ['r', 'dens', 'temp', 'ye', 'entr', 'velx', 'vely', 'lnue', 'lnua', 'lnux', 'lanux', 'enue', 'enua', 'enux', 'eanux', 'ejected']

#wNu = with neutrino info; woNu = without neutrino info
keys_woNu = ['dens', 'temp', 'ye', 'entr', 'velx', 'vely', 'gpot', 'ener']
tracer_entries_woNu = ['r', 'dens', 'temp', 'ye', 'entr', 'velx', 'vely', 'ejected']

if WITH_NEUTRINOS:
    keys = keys_wNu
    tracer_entries = tracer_entries_wNu
else:
    keys = keys_woNu
    tracer_entries = tracer_entries_woNu


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

tracer_entries_fmt={'t': '%.8f',
                    'x': '%.14e',
                    'y': '%.14e',
                    'r': '%.14e', 
                    'dens': '%.14e',
                    'ye' : '%.8f',
                    'temp': '%.14e',
                    'entr': '%.8f',
                    'velx': '%.14e',
                    'vely': '%.14e',
                    'lnue': '%.14e',
                    'lnua': '%.14e',
                    'lnux': '%.14e',
                    'lanux': '%.14e',
                    'enue': '%.14e',
                    'enua': '%.14e',
                    'enux': '%.14e',
                    'eanux': '%.14e',
                    'ejected': '%d'
                    }

# ----------------- SPECIFY COLUMN WIDTH -------------------------------------

def fmt_width(fmt):
    if fmt.endswith('e'):
        return 20
    elif fmt.endswith('f'):
        return 10
    elif fmt.endswith('d'):
        return 1
    else:
        return len(fmt)


#------------------------ CREATING FILES -----------------------------

# --- build header ---
header_cols = []

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

#---------------OPENING FILES AND WRITING HEADER -----------------------------

def _write_one_header(args):
    """
    Write the header of a single tracer file.

    This helper is picklable so it can be used in multiprocessing. 
    Creates a tracer#####.dat file with the initial mass and standard header lines.

    Parameters
    ----------
    args : tuple
        Contains (tracer ID, initial tracer mass, path to tracer files).
    """
    tr_id, init_mass_tr, path_to_tracers, = args
    filename = os.path.join(path_to_tracers, f"tracer{str(tr_id).zfill(5)}.dat")
    with open(filename, "w") as f:
        f.write(f"# Mass of the tracer [g]: {init_mass_tr:.6e}\n")
        f.write(header + "\n")
        f.write(separator + "\n")

def write_all_headers_parallel(init_mass, path_to_tracers, num_cpus):
    """
    Create header files for all tracers in parallel.

    Uses multiprocessing with the specified number of CPUs to speed up
    creation of a large number of tracer#####.dat header files.

    Parameters
    ----------
    init_mass : list or np.ndarray
        Initial mass of each tracer.
    path_to_tracers : str
        Directory to write tracer files.
    num_cpus : int
        Number of worker processes to use.
    """
    ctx = mp.get_context("spawn")          # safe on clusters
    with ctx.Pool(processes=num_cpus) as pool:
        pool.map(_write_one_header,
                 [(tr_id, init_mass[tr_id], path_to_tracers)
                  for tr_id in range(len(init_mass))],
                 chunksize=256)        # keeps overhead low for ≥1e5 files
        

#-------------- ENSURE ASCENDING TIME-ORDER ---------------------------------

def ensure_ascending_time_order_nse_flag(reached_NSE):
    """
    Ensure tracer files have monotonically increasing time.

    Reads each tracer#####.dat file, reverses the data if needed to enforce
    ascending time order (required by WinNet), and keeps the header intact.
    Also appends ', reached_NSE: True/False' to the first header line.

    Parameters
    ----------
    reached_NSE : list or np.ndarray
        Boolean flags indicating whether each tracer reached NSE.
    """
    files = sorted(glob(os.path.join(PATH_TO_OUTPUT, "tracers/tracer*.dat")))


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


# --------------- WRITING CHUNKDATA TO TRACERFILE ---------------------------

def write_to_tracer_file(times, positions, keys, data_dict, tr_id):
    """
    Append one chunk of tracer trajectory data to a tracer#####.dat file.
    
    Handles unit conversions and column ordering required by WinNet.
    Skips the first entry to avoid duplicating data when chaining multiple chunks.
    
    Parameters
    ----------
    times : np.ndarray
        Array of times corresponding to the trajectory chunk.
    positions : tuple of np.ndarray
        (x, y) positions of the tracer for this chunk.
    keys : list of str
        Keys present in data_dict for this chunk.
    data_dict : dict
        Dictionary containing tracer data arrays.
    tr_id : int
        Tracer ID used for the filename.
    """

    x, y = positions
    r = np.sqrt(x**2 + y**2)

    # Prepare dictionary with all data to write
    out = {}
    for entry in tracer_entries:
        if entry in keys:
            # Convert temperature to GK if needed
            out[entry] = data_dict[entry]/1e9 if entry == 'temp' else data_dict[entry]
        elif entry == 'r':
            out[entry] = r / 1e5  # convert from cm to km
        elif entry in ['lnue', 'lnua']:
            # Convert fluxes to WinNet units (erg/s)
            flux_key = 'f' + entry[1:]  # 'lnue' -> 'fnue'
            out[entry] = L_from_f(data_dict[flux_key], r) * 1e51
        elif entry in ['lnux', 'lanux']:
            # Split heavy neutrino flux into two components
            if 'fnux' not in data_dict:
                raise KeyError("Missing flux fnux for lnux/lanux")
            Lx = L_from_f(data_dict['fnux'], r) * 1e51
            out['lnux'] = 0.5 * Lx
            out['lanux'] = 0.5 * Lx
        elif entry == 'eanux':
            if 'enux' not in data_dict:
                raise KeyError("Missing enux for eanux")
            out['eanux'] = data_dict['enux']
        elif entry == 'ejected':
            # Determine if tracer is unbound
            out['ejected'] = is_ejected(
                data_dict['gpot'], data_dict['ener'],
                data_dict['velx'], data_dict['vely'], x, y
            )
        else:
            raise ValueError(f"Don't know how to handle tracer entry {entry}")

    # --- Collect arrays for output ---
    chunk_tracer_dat = [times, x, y]  # base quantities always included
    for entry in tracer_entries:
        if entry in out:
            chunk_tracer_dat.append(out[entry])

    # Stack into 2D array (n_times, n_columns) and skip first row to avoid duplication
    chunk_tracer_dat_array = np.column_stack(chunk_tracer_dat)[1:]

    # --- Prepare formats for saving ---
    entry_fmt_list = [tracer_entries_fmt[entry] for entry in ['t','x','y'] + tracer_entries]

    # --- Write to file ---
    filename = os.path.join(PATH_TO_OUTPUT, f'tracers/tracer{str(tr_id).zfill(5)}.dat')
    with open(filename, 'a') as f:
        np.savetxt(f, chunk_tracer_dat_array, fmt=entry_fmt_list, comments='')
