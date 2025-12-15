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
from time import time
import os, sys, multiprocessing as mp
import multiprocessing as mp
from datetime import datetime

#------- custom modules --------------------------
from config import *
from src.utils import write_log, log_used_params, calc_seeds, SnapshotsCls
from src.tracer_placement import PosFromDens_blockbased, PosFromFile
from src.tracer_integration import sgn, integrate_chunk
from src.tracer_files import ensure_ascending_time_order_nse_flag, write_all_headers_parallel, tracer_entries, keys


if __name__ == "__main__":
    
    t0 = time()
    mp_manager = mp.Manager()

    # -------------- CREATING OUTPUT DIRECTORY ------------------------------
    if not os.path.exists(PATH_TO_OUTPUT):
        os.makedirs(PATH_TO_OUTPUT)
        path_to_tracers = os.path.join(PATH_TO_OUTPUT, 'tracers')
        os.makedirs(path_to_tracers)

    if len(os.listdir(PATH_TO_OUTPUT)) > 1:
        print(f"Directory already exists and is not empty: {PATH_TO_OUTPUT}. Exiting")
        sys.exit(1)

    # ------------- WRITE PARAMETERS TO RUN.LOG --------------------------------
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) #to now which slurm out for which run
    log_used_params() #prints all chosen params in config to run.log

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
    match DIRECTION:
        case 'backward':
            intr_start = SnapshotsCls(PLT_FILES[sgn], keys=keys)
            match PLACEMENT_METHOD:
                case 'PosWithDens':
                    init_x, init_y, init_mass = PosFromDens_blockbased(intr_start, NUM_TRACERS)
                    # init_x, init_y, init_mass = PosFromDens_Bene(intr_start, num_tracers, only_unbound=only_unbound, maxDens=max_dens)
                case 'FromFile':
                    init_x, init_y, init_mass = PosFromFile(PATH_TO_TRACERS_START)
                case _:
                    write_log(PATH_TO_OUTPUT, 'Unknown tracer placement method. Exiting.')
                    sys.exit('Unknown tracer placement method. Exiting.')
        
        case 'forward':
            intr_start = SnapshotsCls(PLT_FILES[0], keys=keys)
            match PLACEMENT_METHOD:
                case 'PosWithDens':
                    init_x, init_y, init_mass = PosFromDens_blockbased(intr_start, NUM_TRACERS)
                case 'FromFile':
                    init_x, init_y, init_mass = PosFromFile(PATH_TO_TRACERS_START)
                case _:
                    write_log(PATH_TO_OUTPUT, 'Unknown tracer placement method. Exiting.')
                    sys.exit('Unknown tracer placement method. Exiting.')
        
        case _:
            write_log(PATH_TO_OUTPUT, 'Unknown integration direction. Exiting.')
            sys.exit('Unknown integration direction. Exiting.')

    initial_pos = np.column_stack((init_x, init_y))

    placed_num_tracers = len(init_mass) #override num_tracers to get real number of placed tracers


    # -------------- SETTING UP SHARED MP-ARRAYS -----------------

    still_calc = mp_manager.Array('b', [True] * placed_num_tracers)  # Shared across all processes
    reached_NSE = mp_manager.Array('b', [False] * placed_num_tracers)
    still_check_NSE = mp_manager.Array('b', [True] * placed_num_tracers)
    chunk_first_steps = mp_manager.Array('d', [1e-6] * placed_num_tracers)
    chunk_first_teval = mp_manager.Array('d', [intr_start.currentSimTime()] * placed_num_tracers)

    t1 = time()

    # --------INITIALIZE AND WRITE HEADERS INTO TRACER FILES -------

    write_log(PATH_TO_OUTPUT, f"Creating tracer files - this will take a while :)")
    
    t_pre_trfile_creation = time()
    write_all_headers_parallel(init_mass, path_to_tracers, num_cpus=num_cpus)

    write_log(PATH_TO_OUTPUT, f"Created all tracer files in {time()-t_pre_trfile_creation:.1f}s")

    # -------------------- EXECUTING MAIN LOOP ---------------------------

    oob_list = []
    failed_list = []

    n_chunks = len(PLT_FILES) // CHUNK_SIZE
    write_log(PATH_TO_OUTPUT,f"Integrating {DIRECTION} in {n_chunks+1} chunks:")
    write_log(PATH_TO_OUTPUT,"____________________________________________________ \n")


    if sgn == 1:
        chunk_ranges = range(0, len(PLT_FILES), CHUNK_SIZE)
    else:
        chunk_ranges = range(len(PLT_FILES) - 1, -1, -CHUNK_SIZE)

    pos = initial_pos
    prev_end_index = None  # track previous chunk boundary

    for start in chunk_ranges:
        if sgn == 1:
            end = min(start + CHUNK_SIZE, len(PLT_FILES))

            # overlap: start one before if not the first chunk
            if prev_end_index is not None:
                plt_chunk = PLT_FILES[prev_end_index - 1:end]
            else:
                plt_chunk = PLT_FILES[start:end]

        else:  # backward integration
            end = max(start - CHUNK_SIZE + 1, 0)

            if prev_end_index is not None:
                plt_chunk = PLT_FILES[end:prev_end_index + 1][::-1]
            else:
                plt_chunk = PLT_FILES[end:start + 1][::-1]

        prev_end_index = end

        if len(plt_chunk) == 1:
            write_log(PATH_TO_OUTPUT, f'Chunk starting at {start} has only one snapshot in it - ignore it')
            continue

        write_log(PATH_TO_OUTPUT,f'Starting chunk from index {start} to {end}')
        active_before = sum(still_calc)

        MP_arrays = [still_calc, reached_NSE, still_check_NSE, chunk_first_steps, chunk_first_teval]

        chunk_args =   (plt_chunk, pos,  tracer_entries, [RTOL, ATOL, MAXSTEP], MP_arrays,
                        PATH_TO_OUTPUT, TIME_LIMIT, keys, num_cpus)
        pos, chunk_OOB_events, chunk_failed_events = integrate_chunk(chunk_args=chunk_args)
        active_after = sum(still_calc)
        oob_list.extend(chunk_OOB_events)
        failed_list.extend(chunk_failed_events)
        write_log(PATH_TO_OUTPUT,f"Active tracers: {active_after}\n")

    #------------ CLEAN UP AFTER INTEGRATION ---------------------

    write_log(PATH_TO_OUTPUT,f'{len(oob_list)} OOB events:')
    
    #print all oob events
    for oob_event in oob_list:
        write_log(PATH_TO_OUTPUT,f'  {oob_event}')

    write_log(PATH_TO_OUTPUT,f'{len(failed_list)} failed Tracers:')

    #print all failed events
    for fail_event in failed_list:
        write_log(PATH_TO_OUTPUT, f'    {fail_event}')

    write_log(PATH_TO_OUTPUT,f"Integration complete. Total timesteps: {len(PLT_FILES)}")

    #if backwards - reverse time order in tracer file to ascending
    if DIRECTION == 'backward':
        write_log(PATH_TO_OUTPUT, f'Assuring ascending time order for nuclear network')
        ensure_ascending_time_order_nse_flag(reached_NSE)

    #if calc_seeds - calculate initial composition of the tracers from porgenitor file
    if CALC_SEEDS:
        write_log(PATH_TO_OUTPUT, f'Starting to calculate initial compositions of tracers')
        calc_seeds()

    write_log(PATH_TO_OUTPUT, f'Tracer calculation complete :) - total time {(time()-t0)/60:.2f} minutes.')
