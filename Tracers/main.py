# ==============================================================================
#  2-D FLASH TRACER INTEGRATION
#  Author: Benedikt Weinhold  (~Oct 2025)
#
#  MAIN
#
#  main execution of the program depeding on the input in config
#  For further explanations see README.md 
# ==============================================================================

# ---------- standard library -------------------------------
import numpy as np
from glob import glob
from time import time
import os, sys, multiprocessing as mp
import multiprocessing as mp
from datetime import datetime

#------- custom modules --------------------------

from config import *
from src.utils import write_log
from src.tracer_files import keys 
from src.tracer_placement import PosFromDens_Bene_steps, PosFromFile
from src.tracer_integration import sgn, integrate_chunk
from src.tracer_files import ensure_ascending_time_order_nse_flag, write_all_headers_parallel, fmt_width, tracer_entries_fmt, tracer_entries_units, tracer_entries, keys

import src.Progenitors as Prog
import src.Snapshot2D as Snap



if __name__ == "__main__":
    
    t0 = time()
    mp_manager = mp.Manager()

    # -------------- CREATING OUTPUT DIRECTORY ------------------------------
    if not os.path.exists(PATH_TO_OUTPUT):
        os.makedirs(PATH_TO_OUTPUT)

    if len(os.listdir(PATH_TO_OUTPUT)) > 1:
        print(f"Directory already exists and is not empty: {PATH_TO_OUTPUT}. Exiting")
        sys.exit(1)

    # ------------- WRITE PARAMETERS TO RUN.LOG --------------------------------
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) #to now which slurm out for which run
    write_log(PATH_TO_OUTPUT, f"\n")
    write_log(PATH_TO_OUTPUT, f"Starting tracer calculation...")
    write_log(PATH_TO_OUTPUT, f"Chosen Parameters: ")
    write_log(PATH_TO_OUTPUT, f"       Integration direction = {DIRECTION}")
    write_log(PATH_TO_OUTPUT, f"       calc_seeds = {CALC_SEEDS}")
    write_log(PATH_TO_OUTPUT, f"       with_neutrinos = {WITH_NEUTRINOS}")
    write_log(PATH_TO_OUTPUT, f"       Tracer placement = {PLACEMENT_METHOD}")

    if PLACEMENT_METHOD == 'PosWithDens':
        write_log(PATH_TO_OUTPUT, f"       num_tracers = {NUM_TRACERS}")
        write_log(PATH_TO_OUTPUT, f"       max_temp_place = {MAX_TEMP_PLACE}")
        write_log(PATH_TO_OUTPUT, f"       only_unbound = {ONLY_UNBOUND}")
        if ONLY_UNBOUND==False:
            write_log(PATH_TO_OUTPUT, f"       max_dens = {MAX_DENS:.1e}")
    
    write_log(PATH_TO_OUTPUT, f'       only_until_maxTemp = {ONLY_UNTIL_MAXTEMP}')
    if ONLY_UNTIL_MAXTEMP:
        write_log(PATH_TO_OUTPUT, f'       max_temp = {MAXTEMP_TRACER/1e9:.2e} GK')

    write_log(PATH_TO_OUTPUT,f"       used tolerances: rtol= {RTOL:.1e}, atol={ATOL:.1e}, maxstep={MAXSTEP:.1e}")
    write_log(PATH_TO_OUTPUT, f"\n")

    if ARB_MESSAGE: #check to see its not empty e.g. ""
        write_log(PATH_TO_OUTPUT, 'User message: '+ ARB_MESSAGE)


    #----------------- GET RESOURCES FROM SLURM -------------
    try:
        num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK'))
        if num_cpus is None:
            raise ValueError
    except (TypeError, ValueError):
        # Fallback if not running under Slurm or variable missing
        num_cpus = 1

    write_log(PATH_TO_OUTPUT, f"Using {num_cpus} CPUs")


    # ----------- PLACE TRACERS ACCORDING TO METHOD AND DIRECTION -----------


    if DIRECTION == 'backward':
        intr_start = Snap.Snapshot2D(PLT_FILES[sgn], keys=keys)
        if PLACEMENT_METHOD == 'PosWithDens':
            init_x, init_y, init_mass = PosFromDens_Bene_steps(intr_start, NUM_TRACERS, only_unbound=ONLY_UNBOUND, maxDens=MAX_DENS, maxTemp=MAX_TEMP_PLACE)
            # init_x, init_y, init_mass = PosFromDens_Bene(intr_start, num_tracers, only_unbound=only_unbound, maxDens=max_dens)
        elif PLACEMENT_METHOD == 'FromFile':
            init_x, init_y, init_mass = PosFromFile(PATH_TO_TRACERS_START)
        else:
            write_log(PATH_TO_OUTPUT, 'Unknown tracer placement method. Exiting.')
            sys.exit('Unknown tracer placement method. Exiting.')   
    elif DIRECTION == 'forward':
        intr_start = Snap.Snapshot2D(PLT_FILES[0], keys=keys)
        if PLACEMENT_METHOD == 'PosWithDens':
            init_x, init_y, init_mass = PosFromDens_Bene_steps(intr_start, NUM_TRACERS, only_unbound=ONLY_UNBOUND, maxDens=MAX_DENS, maxTemp=MAX_TEMP_PLACE)
        elif PLACEMENT_METHOD == 'FromFile':
            init_x, init_y, init_mass = PosFromFile(PATH_TO_TRACERS_START)
        else:
            write_log(PATH_TO_OUTPUT, 'Unknown tracer placement method. Exiting.')
            sys.exit('Unknown tracer placement method. Exiting.')
    else:
        write_log(PATH_TO_OUTPUT, 'Unknown integration direction. Exiting.')
        sys.exit('Unknown integration direction. Exiting.')

    initial_pos = np.column_stack((init_x, init_y))

    placed_num_tracers = len(init_mass) #override num_tracers to get real number of placed tracers


    # -------------- SETTING UP SHARED MP-ARRAYS -----------------

    still_calc = mp_manager.Array('b', [True] * placed_num_tracers)  # Shared across all processes
    reached_NSE = mp_manager.Array('b', [False] * placed_num_tracers)
    still_check_NSE = mp_manager.Array('b', [True] * placed_num_tracers)

    t1 = time()

    # --------INITIALIZE AND WRITE HEADERS INTO TRACER FILES -------

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

    write_log(PATH_TO_OUTPUT, f"Starting to write the tracer files - this will take a while :)")
    
    write_all_headers_parallel(init_mass, PATH_TO_OUTPUT, header, separator, num_cpus=num_cpus)


    # -------------------- EXECUTING MAIN LOOP ---------------------------

    oob_list = []

    n_chunks = len(PLT_FILES) // CHUNK_SIZE
    write_log(PATH_TO_OUTPUT,f"Integrating {DIRECTION} in {n_chunks} chunks:")
    write_log(PATH_TO_OUTPUT,"____________________________________________________ \n")

    if sgn == 1:
        chunk_ranges = range(0, len(PLT_FILES), CHUNK_SIZE)
    else:
        chunk_ranges = range(len(PLT_FILES) -1 , -1, -CHUNK_SIZE)

    pos = initial_pos

    for start in chunk_ranges:
        if sgn == 1:
            end = min(start + CHUNK_SIZE, len(PLT_FILES))
            plt_chunk = PLT_FILES[start:end]
        else:
            end = max(start - CHUNK_SIZE + 1, 0)
            plt_chunk = PLT_FILES[end:start + 1][::-1]

        if len(plt_chunk) == 1:
            write_log(PATH_TO_OUTPUT, f'Chunk starting at {start} has only one snapshot in it - ignore it')
            continue

        write_log(PATH_TO_OUTPUT,f'Starting chunk from index {start} to {end}')
        active_before = sum(still_calc)
        chunk_args =   (plt_chunk, pos,  tracer_entries, [RTOL, ATOL, MAXSTEP], still_calc, reached_NSE,
                        still_check_NSE, PATH_TO_OUTPUT, tracer_entries_fmt, TIME_LIMIT, keys, num_cpus)
        pos = integrate_chunk(chunk_args=chunk_args)
        active_after = sum(still_calc)
        write_log(PATH_TO_OUTPUT,f"Active tracers after chunk: {active_after}\n")

    write_log(PATH_TO_OUTPUT,f'{len(oob_list)} OOB events:')

    for oob_event in oob_list:
        write_log(PATH_TO_OUTPUT,f'  {oob_event}')

    write_log(PATH_TO_OUTPUT,f"Integration complete. Total timesteps: {len(PLT_FILES)}")

    if DIRECTION == 'backward':
        write_log(PATH_TO_OUTPUT, f'Assuring ascending time order for nuclear network')
        ensure_ascending_time_order_nse_flag(PATH_TO_OUTPUT, reached_NSE)

    if CALC_SEEDS:
        write_log(PATH_TO_OUTPUT, f'Starting to calculate initial compositions of tracers')
        seeds_dir = os.path.join(PATH_TO_OUTPUT, 'seeds')
        if not os.path.exists(seeds_dir):
            os.makedirs(seeds_dir)  # create the seeds directory
        
        tracers = sorted(glob(os.path.join(PATH_TO_OUTPUT, "tracer*")))

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

    write_log(PATH_TO_OUTPUT, f'Tracer calculation complete :) - total time {(time()-t0)/60:.2f} minutes.')
