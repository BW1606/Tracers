# ==============================================================================
#  2-D FLASH TRACER INTEGRATION
#  Author: Benedikt Weinhold  (~Oct 2025)
#
#  TRACER_INTEGRATION
#
#  integrate_chunk: Sets up snapshot chunk, loads into shared memory, and
#                   spawns multiprocessing workers
#
#  integrate_single_tracer: Integrates single tracer through simulation chunk
#                           using solve_ivp (RKF45 solver)
#
#  For further explanations see README.md 
# ==============================================================================

import traceback, time
import numpy as np
import multiprocessing as mp
import threading
import time
from scipy.integrate import solve_ivp
import psutil
import os

from .utils import write_log, cleanup_shared_memory, create_progress_bar, load_snapshots_into_shm, reconstruct_snapshots
from .tracer_files import write_to_tracer_file, keys
from config import PATH_TO_OUTPUT, MAXTEMP_TRACER, NSE_TEMP, ONLY_UNTIL_MAXTEMP, XMIN, XMAX, YMAX, YMIN, DIRECTION, CHUNK_SIZE, LOG_EVERY


# ==============================================================================
#  TRACER INTEGRATION (chunk setup)
# ==============================================================================

# Set integration direction
if DIRECTION == 'forward':
    sgn = 1
elif DIRECTION == 'backward':
    sgn = -1
else:
    write_log(PATH_TO_OUTPUT, "Unknown integration direction")

def _integrate_single_tracer_wrapper(args):
    """Unpacks tuple arguments for imap_unordered compatibility"""
    return integrate_single_tracer(*args)

def integrate_chunk(chunk_args):
    """
    Integrates a chunk of tracers through 2D FLASH simulation snapshots.

    Steps:
    1. Loads snapshots into shared memory for efficient parallel access
    2. Integrates each active tracer in parallel using multiprocessing
    3. Cleans up shared memory when done

    Args:
        chunk_args (tuple): Contains all required arguments for this chunk:
            - plt_files_chunk: List of snapshot file paths
            - startpos: Initial positions of tracers
            - tracer_entries: Array storing tracer data
            - solve_ivp_args: Arguments for solver
            - MP_arrays: Multiprocessing arrays (still_calc, reached_NSE, etc.)
            - output_dir: Path to write logs
            - time_limit: Max allowed time for integration
            - td_vars_keys: Keys of thermodynamic variables to extract
            - num_cpus: Number of parallel processes to spawn
            
    Returns:
        tuple: (new_positions, oob_events, failed_events)
    """
    # Unpack arguments
    (plt_files_chunk, startpos, tracer_entries, solve_ivp_args, MP_arrays, 
     output_dir, time_limit, td_vars_keys, num_cpus) = chunk_args

    # Initialize storage for snapshot metadata and shared memory handles
    snapshots_meta = []
    times_chunk = []
    shm_handles_all = []

    def integrate_tracers_parallel():
        """
        Integrates all active tracers in parallel using multiprocessing.
        Logs progress and returns new positions and event lists.
        """
        # Get active tracer IDs
        active_tracer_ids = np.where(MP_arrays[0])[0]
        n_tasks = len(active_tracer_ids)
        new_positions = np.full((len(startpos), 2), np.nan)
        oob_list, failed_list = [], []

        write_log(output_dir, f"Starting parallel integration of {n_tasks} tracers...")

        # Prepare argument tuples for each tracer
        task_args = [
            (
                tr_id,
                startpos[tr_id],
                times_chunk,
                snapshots_meta,
                sgn,
                keys,
                tracer_entries,
                solve_ivp_args,
                MP_arrays,
                output_dir,
                time_limit
            )
            for tr_id in active_tracer_ids
        ]

        # Chunk size for imap_unordered
        chunksize = max(1, n_tasks // (num_cpus * 4))

        # --- HEARTBEAT LOGGER SETUP ---
        start_time = time.time()
        LOG_INTERVAL = LOG_EVERY
        progress_counter = {"i": 0}   # Mutable so heartbeat thread can read it
        stop_flag = {"stop": False}

        def heartbeat():
            """Logs progress every LOG_INTERVAL seconds independently of task completion."""
            while not stop_flag["stop"]:
                time.sleep(LOG_INTERVAL)
                i = progress_counter["i"]
                progress_bar = create_progress_bar(i, n_tasks)
                write_log(
                    output_dir,
                    f"     Chunk progress : {progress_bar}"
                )

        # Start heartbeat thread
        hb_thread = threading.Thread(target=heartbeat, daemon=True)
        hb_thread.start()

        try:
            # Process tracers in parallel
            with mp.Pool(processes=num_cpus) as pool:
                for i, res in enumerate(
                    pool.imap_unordered(_integrate_single_tracer_wrapper, task_args, chunksize=chunksize),
                    start=1
                ):
                    # Update progress for heartbeat thread
                    progress_counter["i"] = i

                    # --- Process results ---
                    if isinstance(res, Exception):
                        failed_list.append(("unknown", repr(res)))
                    else:
                        tr_pos, oob_events, fail_events, tr_id = res
                        new_positions[tr_id] = tr_pos
                        oob_list.extend(oob_events)
                        failed_list.extend(fail_events)

        except Exception as e:
            err_msg = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            write_log(output_dir, f"[ERROR] Exception in parallel pool: {err_msg}")
            raise

        finally:
            # Stop heartbeat and wait for it
            stop_flag["stop"] = True
            hb_thread.join(timeout=1)

        write_log(output_dir, "All tracer integrations completed successfully.")
        return new_positions, oob_list, failed_list

    # def integrate_tracers_parallel():
    #     """
    #     Integrates all active tracers in parallel using multiprocessing.
    #     Logs progress and returns new positions and event lists.
    #     """
    #     # Get active tracer IDs
    #     active_tracer_ids = np.where(MP_arrays[0])[0]  # MP_arrays[0] is still_calc
    #     n_tasks = len(active_tracer_ids)
    #     new_positions = np.full((len(startpos), 2), np.nan)
    #     oob_list, failed_list = [], []

    #     write_log(output_dir, f"Starting parallel integration of {n_tasks} tracers...")

    #     # Prepare argument tuples for each tracer
    #     task_args = [
    #         (
    #             tr_id,
    #             startpos[tr_id],
    #             times_chunk,
    #             snapshots_meta,
    #             sgn,
    #             keys,
    #             tracer_entries,
    #             solve_ivp_args,
    #             MP_arrays,
    #             output_dir,
    #             time_limit
    #         )
    #         for tr_id in active_tracer_ids
    #     ]

    #     # Calculate logging interval (log 10 times max)
    #     chunksize = max(1, n_tasks // (num_cpus * 4))
        
    #     #TODO: check if this logging works 
    #     start_time = time.time()
    #     last_log = start_time
    #     log_interval_sec = LOG_EVERY

    #     try:
    #         # Process tracers in parallel
    #         with mp.Pool(processes=num_cpus) as pool:
    #             for i, res in enumerate(
    #                 pool.imap_unordered(_integrate_single_tracer_wrapper, task_args, chunksize=chunksize),
    #                 start=1
    #             ):
    #                 # --- Process results ---
    #                 if isinstance(res, Exception):
    #                     failed_list.append(("unknown", repr(res)))
    #                 else:
    #                     tr_pos, oob_events, fail_events, tr_id = res
    #                     new_positions[tr_id] = tr_pos
    #                     oob_list.extend(oob_events)
    #                     failed_list.extend(fail_events)

    #                 # --- Time-based progress logging ---
    #                 now = time.time()
    #                 if now - last_log >= log_interval_sec or i == n_tasks:
    #                     elapsed = now - start_time
    #                     progress_bar = create_progress_bar(i, n_tasks)
    #                     write_log(
    #                         output_dir,
    #                         f"     Chunk progress: {progress_bar}"
    #                     )
    #                     last_log = now

    #     except Exception as e:
    #         err_msg = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
    #         write_log(output_dir, f"[ERROR] Exception in parallel pool: {err_msg}")
    #         raise

    #     write_log(output_dir, "All tracer integrations completed successfully.")
    #     return new_positions, oob_list, failed_list

    try:
        active_tracer_number = len(np.where(MP_arrays[0])[0])
        
        # Load snapshots into shared memory
        snapshots_meta, times_chunk, shm_handles_all = load_snapshots_into_shm(plt_files_chunk, td_vars_keys, sgn)
        
        # Integrate active tracers
        t1 = time.time()
        new_positions, oob_events, failed_events = integrate_tracers_parallel()
        calc_time_chunk = time.time() - t1

        write_log(output_dir, f"Integrated chunk in {calc_time_chunk:.2f}s, avg. time per tracer per snapshot: {calc_time_chunk/(CHUNK_SIZE * active_tracer_number):.4f}s")

        # Return final positions and events
        return np.array(new_positions), oob_events, failed_events

    except Exception as e:
        print("❌ Exception in integrate_chunk:", e)
        traceback.print_exc()
        raise

    finally:
        # Always clean up shared memory
        cleanup_shared_memory(shm_handles_all)


# ==============================================================================
#  TRACER INTEGRATION (single tracer)
# ==============================================================================

# def integrate_single_tracer(tr_id, start_pos, times_chunk, snapshots_meta, sgn, keys,
#                             tracer_entries, solve_ivp_args, MP_arrays, output_dir,
#                             time_limit):
#     """
#     Integrates a single tracer through the FLASH 2D velocity field.
    
#     Uses shared-memory snapshots (no file I/O inside worker).
#     Returns final position and any out-of-bounds (OOB) events.
#     """
#     still_calc, reached_NSE, still_check_NSE, chunk_first_steps, chunk_first_teval = MP_arrays
#     local_oob_events = []
#     tracer_start_time = time.time()
#     t_eval = times_chunk

#     def velocity_field(t, pos):
#         """
#         Interpolates velocity field at position (x, y) and time t.
#         Includes timeout protection and forward/backward integration handling.
#         """
#         x, y = pos

#         # Check for wall-clock timeout
#         if (time.time() - tracer_start_time) > time_limit:
#             msg = f'Tracer {tr_id}: timeout ({time_limit:.1f}s) at t={t:.5f}s, pos=({x/1e5:.6e},{y/1e5:.6e}) km'
#             write_log(output_dir, msg)
#             raise RuntimeError(msg)

#         # Normalize to ascending time order
#         if sgn == 1:  # Forward integration
#             times_asc = times_chunk
#             snaps_ordered = snapshots
#         else:  # Backward integration
#             times_asc = times_chunk[::-1]
#             snaps_ordered = snapshots[::-1]

#         # Handle boundary cases
#         if t <= times_asc[0]:
#             idx = 0
#         elif t >= times_asc[-1]:
#             idx = -1
#         else:
#             # Interpolate between bracketing snapshots
#             idx_r = np.searchsorted(times_asc, t, side='right')
#             idx_l = idx_r - 1
            
#             # Get velocities at bracketing times
#             t_pair = [times_asc[idx_l], times_asc[idx_r]]
#             velx_pair = [snaps_ordered[idx_l].getQuantAtPos('velx', x, y, tr_id),
#                         snaps_ordered[idx_r].getQuantAtPos('velx', x, y, tr_id)]
#             vely_pair = [snaps_ordered[idx_l].getQuantAtPos('vely', x, y, tr_id),
#                         snaps_ordered[idx_r].getQuantAtPos('vely', x, y, tr_id)]
            
#             return [np.interp(t, t_pair, velx_pair), 
#                     np.interp(t, t_pair, vely_pair)]

#         # Return velocity at nearest snapshot
#         return [snaps_ordered[idx].getQuantAtPos('velx', x, y, tr_id),
#                 snaps_ordered[idx].getQuantAtPos('vely', x, y, tr_id)]

#     def oob_event(t, pos):
#         """Detects when tracer leaves simulation domain"""
#         x, y = pos
#         return min(x, XMAX-x, y-YMIN, YMAX-y)  # Negative if outside domain

#     oob_event.terminal = True
#     oob_event.direction = -1  # Triggers when tracer leaves domain

#     def sample_vars(positions, keys, snapshots, t_eval):
#         """Samples thermodynamic variables at tracer positions"""
#         sampled = {key: np.empty(len(t_eval)) for key in keys}
#         for i, t in enumerate(t_eval):
#             x = positions[0, i]
#             y = positions[1, i]
#             snap = snapshots[i]
#             print(f't_eval t: {t}, snapshot t: {snap.simtime}')
#             for key in keys:
#                 sampled[key][i] = snap.getQuantAtPos(key, x, y, tr_id)
#         return sampled

#     try:
#         # Reconstruct snapshots from shared memory
#         snapshots = reconstruct_snapshots(snapshots_meta)
#         t_reconstruct = time.time()

#         # Set integration span and initial position
#         t_span = [chunk_first_teval[tr_id], times_chunk[-1]]
#         start_pos = [start_pos[0], start_pos[1]]

#         # Integrate tracer trajectory
#         result = solve_ivp(
#             velocity_field,
#             t_span=t_span,
#             y0=start_pos,
#             rtol=solve_ivp_args[0],
#             atol=solve_ivp_args[1],
#             max_step=solve_ivp_args[2],
#             method='RK45',
#             dense_output=True,
#             first_step=chunk_first_steps[tr_id],
#             events=oob_event
#         )

#         pos_eval = result.sol(t_eval)
#         t_post_integration = time.time()

#         # Handle out-of-bounds events
#         if result.t_events[0].size > 0:
#             still_calc[tr_id] = False
#             exit_time = result.t_events[0][0]
#             exit_pos = result.y_events[0][0]
#             log_pos = (f"{np.round(exit_pos[0]/1e5,2)}, {np.round(exit_pos[1]/1e5,2)}"
#                        if exit_pos is not None else "unknown")
#             local_oob_events.append(f"  Tracer {tr_id} OOB at t={np.round(exit_time,3)}, pos=({log_pos}) km")

#             # Sample and write in-bounds variables
#             vars_sampled = sample_vars(pos_eval, keys=keys, snapshots=snapshots, t_eval=t_eval)
#             write_to_tracer_file(t_eval, pos_eval, keys, vars_sampled, tracer_entries, tr_id, output_dir)
#             return np.array([result.y[0][-1], result.y[1][-1]]), local_oob_events, [], tr_id

#         # Sample variables at evaluation points
#         vars_sampled = sample_vars(positions=pos_eval, keys=keys, snapshots=snapshots, t_eval=t_eval)

#         # Handle NSE conditions for backward tracers
#         if sgn == -1 and still_check_NSE and 'temp' in vars_sampled:
#             if max(vars_sampled['temp']) > NSE_TEMP:
#                 reached_NSE[tr_id] = True
#                 still_check_NSE[tr_id] = False
#                 if ONLY_UNTIL_MAXTEMP and max(vars_sampled['temp']) > MAXTEMP_TRACER:
#                     still_calc[tr_id] = False
#                     write_log(output_dir, f"Tracer {tr_id} reached max_temp and will be terminated")

#         # Write tracer data to file
#         write_to_tracer_file(t_eval, pos_eval, keys, vars_sampled, tracer_entries, tr_id, output_dir)
#         t_post_writing = time.time()

#         # Log timing information for selected tracers
#         # if tr_id % 1000 == 0:
#         #     write_log(output_dir, f'    Tracer {tr_id}: snap-reconstruction: {t_reconstruct - tracer_start_time:.2f}s, \
#         #             integration in  {t_post_integration - t_reconstruct:.2f}s \
#         #             I/O saving in {t_post_writing - t_post_integration:.2f}s\
#         #             overall: {t_post_writing - tracer_start_time:.2f}s')

#         # Update multiprocessing arrays for next chunk
#         chunk_first_teval[tr_id] = t_eval[-1]
#         chunk_first_steps[tr_id] = abs(result.t[-1] - result.t[-2])

#         return np.array([pos_eval[0][-1], pos_eval[1][-1]]), local_oob_events, [], tr_id

#     except Exception as e:
#         tb_str = traceback.format_exc()
#         write_log(output_dir, f"❌ Tracer {tr_id} failed: {e}, tb: {tb_str}")
#         still_calc[tr_id] = False
#         return np.array([np.nan, np.nan]), [], [f"❌ Tracer {tr_id} failed: {e}"], tr_id

#     finally:
#         # Always clean up shared memory
#         for snap in snapshots:
#             snap.close_shm()

def integrate_single_tracer(tr_id, start_pos, times_chunk, snapshots_meta, sgn, keys,
                            tracer_entries, solve_ivp_args, MP_arrays, output_dir,
                            time_limit):
    """
    Integrates a single tracer through the FLASH 2D velocity field.
    
    Uses shared-memory snapshots (no file I/O inside worker).
    Returns final position and any out-of-bounds (OOB) events.
    """
    # Track memory at start
    process = psutil.Process(os.getpid())
    pid = os.getpid()
    mem_start = process.memory_info().rss / 1024**2  # MB
    
    still_calc, reached_NSE, still_check_NSE, chunk_first_steps, chunk_first_teval = MP_arrays
    local_oob_events = []
    tracer_start_time = time.time()
    t_eval = times_chunk

    def velocity_field(t, pos):
        """
        Interpolates velocity field at position (x, y) and time t.
        Includes timeout protection and forward/backward integration handling.
        """
        x, y = pos

        # Check for wall-clock timeout
        if (time.time() - tracer_start_time) > time_limit:
            msg = f'Tracer {tr_id}: timeout ({time_limit:.1f}s) at t={t:.5f}s, pos=({x/1e5:.6e},{y/1e5:.6e}) km'
            write_log(output_dir, msg)
            raise RuntimeError(msg)

        # Normalize to ascending time order
        if sgn == 1:  # Forward integration
            times_asc = times_chunk
            snaps_ordered = snapshots
        else:  # Backward integration
            times_asc = times_chunk[::-1]
            snaps_ordered = snapshots[::-1]

        # Handle boundary cases
        if t <= times_asc[0]:
            idx = 0
        elif t >= times_asc[-1]:
            idx = -1
        else:
            # Interpolate between bracketing snapshots
            idx_r = np.searchsorted(times_asc, t, side='right')
            idx_l = idx_r - 1
            
            # Get velocities at bracketing times
            t_pair = [times_asc[idx_l], times_asc[idx_r]]
            velx_pair = [snaps_ordered[idx_l].getQuantAtPos('velx', x, y, tr_id),
                        snaps_ordered[idx_r].getQuantAtPos('velx', x, y, tr_id)]
            vely_pair = [snaps_ordered[idx_l].getQuantAtPos('vely', x, y, tr_id),
                        snaps_ordered[idx_r].getQuantAtPos('vely', x, y, tr_id)]
            
            return [np.interp(t, t_pair, velx_pair), 
                    np.interp(t, t_pair, vely_pair)]

        # Return velocity at nearest snapshot
        return [snaps_ordered[idx].getQuantAtPos('velx', x, y, tr_id),
                snaps_ordered[idx].getQuantAtPos('vely', x, y, tr_id)]

    def oob_event(t, pos):
        """Detects when tracer leaves simulation domain"""
        x, y = pos
        return min(x, XMAX-x, y-YMIN, YMAX-y)  # Negative if outside domain

    oob_event.terminal = True
    oob_event.direction = -1  # Triggers when tracer leaves domain

    def sample_vars(positions, keys, snapshots, t_eval):
        """Samples thermodynamic variables at tracer positions"""
        sampled = {key: np.empty(len(t_eval)) for key in keys}
        for i, t in enumerate(t_eval):
            x = positions[0, i]
            y = positions[1, i]
            snap = snapshots[i]
            #print(f't_eval t: {t}, snapshot t: {snap.simtime}')
            for key in keys:
                sampled[key][i] = snap.getQuantAtPos(key, x, y, tr_id)
        return sampled

    try:
        # Reconstruct snapshots from shared memory
        snapshots = reconstruct_snapshots(snapshots_meta)
        t_reconstruct = time.time()
        
        # Check memory after snapshot reconstruction
        mem_after_reconstruct = process.memory_info().rss / 1024**2  # MB

        # Set integration span and initial position
        t_span = [chunk_first_teval[tr_id], times_chunk[-1]]
        start_pos = [start_pos[0], start_pos[1]]

        # Integrate tracer trajectory
        result = solve_ivp(
            velocity_field,
            t_span=t_span,
            y0=start_pos,
            rtol=solve_ivp_args[0],
            atol=solve_ivp_args[1],
            max_step=solve_ivp_args[2],
            method='RK45',
            dense_output=True,
            first_step=chunk_first_steps[tr_id],
            events=oob_event
        )

        pos_eval = result.sol(t_eval)
        t_post_integration = time.time()

        # Handle out-of-bounds events
        if result.t_events[0].size > 0:
            still_calc[tr_id] = False
            exit_time = result.t_events[0][0]
            exit_pos = result.y_events[0][0]
            log_pos = (f"{np.round(exit_pos[0]/1e5,2)}, {np.round(exit_pos[1]/1e5,2)}"
                       if exit_pos is not None else "unknown")
            local_oob_events.append(f"  Tracer {tr_id} OOB at t={np.round(exit_time,3)}, pos=({log_pos}) km")

            # Sample and write in-bounds variables
            vars_sampled = sample_vars(pos_eval, keys=keys, snapshots=snapshots, t_eval=t_eval)
            write_to_tracer_file(t_eval, pos_eval, keys, vars_sampled, tr_id)
            
            # Final memory check before return
            mem_end = process.memory_info().rss / 1024**2  # MB
            # write_log(output_dir, 
            #           f"Tracer {tr_id} (PID {pid}) MEMORY: "
            #           f"Start={mem_start:.1f}MB, "
            #           f"AfterReconstruct={mem_after_reconstruct:.1f}MB, "
            #           f"End={mem_end:.1f}MB, "
            #           f"Delta={mem_end-mem_start:.1f}MB")
            
            return np.array([result.y[0][-1], result.y[1][-1]]), local_oob_events, [], tr_id

        # Sample variables at evaluation points
        vars_sampled = sample_vars(positions=pos_eval, keys=keys, snapshots=snapshots, t_eval=t_eval)

        # Handle NSE conditions for backward tracers
        if sgn == -1 and still_check_NSE and 'temp' in vars_sampled:
            if max(vars_sampled['temp']) > NSE_TEMP:
                reached_NSE[tr_id] = True
                still_check_NSE[tr_id] = False
                if ONLY_UNTIL_MAXTEMP and max(vars_sampled['temp']) > MAXTEMP_TRACER:
                    still_calc[tr_id] = False
                    write_log(output_dir, f"Tracer {tr_id} reached max_temp and will be terminated")

        # Write tracer data to file
        write_to_tracer_file(t_eval, pos_eval, keys, vars_sampled, tr_id)
        t_post_writing = time.time()

        # Final memory check
        mem_end = process.memory_info().rss / 1024**2  # MB
        
        # Log timing and memory information
        # write_log(output_dir, 
        #           f"Tracer {tr_id} (PID {pid}) MEMORY: "
        #           f"Start={mem_start:.1f}MB, "
        #           f"AfterReconstruct={mem_after_reconstruct:.1f}MB, "
        #           f"End={mem_end:.1f}MB, "
        #           f"Delta={mem_end-mem_start:.1f}MB")

        # Log timing information for selected tracers
        # if tr_id % 1000 == 0:
        #     write_log(output_dir, f'    Tracer {tr_id}: snap-reconstruction: {t_reconstruct - tracer_start_time:.2f}s, \
        #             integration in  {t_post_integration - t_reconstruct:.2f}s \
        #             I/O saving in {t_post_writing - t_post_integration:.2f}s\
        #             overall: {t_post_writing - tracer_start_time:.2f}s')

        # Update multiprocessing arrays for next chunk
        chunk_first_teval[tr_id] = t_eval[-1]
        chunk_first_steps[tr_id] = abs(result.t[-1] - result.t[-2])

        return np.array([pos_eval[0][-1], pos_eval[1][-1]]), local_oob_events, [], tr_id

    except Exception as e:
        tb_str = traceback.format_exc()
        write_log(output_dir, f"❌ Tracer {tr_id} failed: {e}, tb: {tb_str}")
        still_calc[tr_id] = False
        
        # Memory check on error
        # mem_end = process.memory_info().rss / 1024**2  # MB
        # write_log(output_dir, 
        #           f"Tracer {tr_id} (PID {pid}) MEMORY AT ERROR: "
        #           f"Start={mem_start:.1f}MB, End={mem_end:.1f}MB, Delta={mem_end-mem_start:.1f}MB")
        
        return np.array([np.nan, np.nan]), [], [f"❌ Tracer {tr_id} failed: {e}"], tr_id

    finally:
        # Always clean up shared memory
        for snap in snapshots:
            snap.close_shm()