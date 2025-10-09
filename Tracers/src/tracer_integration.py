# ==============================================================================
#  2-D FLASH TRACER INTEGRATION
#  Author: Benedikt Weinhold  (~Oct 2025)
#
#  TRACER_INTEGRATION
#
#  integrate_chunk: setup chunk of snapshots, read in and store in shared memory
#                   &  set up multiprocessing workers
#
#  integrate_single_tracer: integrates single tracer throughout Simulation chunk
#                           using solve_ivp (wrapper for a RKF45)
#
#  For further explanations see README.md 
# ==============================================================================

import os, psutil, traceback, time
import os
import numpy as np
import multiprocessing as mp
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from .utils import write_log, create_shm_for_array, create_shm_for_snapshot, cleanup_shared_memory
from .tracer_files import write_to_tracer_file, keys, tracer_entries
from config import PATH_TO_OUTPUT, MAXTEMP_TRACER, NSE_TEMP, ONLY_UNTIL_MAXTEMP, XMIN, XMAX, YMAX, YMIN, DIRECTION, WITH_NEUTRINOS

import src.Snapshot2D as Snap

# ==============================================================================
#  TRACER INTEGRATION  (set up chunk)
# ==============================================================================

if DIRECTION == 'forward':
    sgn = 1
elif DIRECTION == 'backward':
    sgn = -1
else:
    write_log(PATH_TO_OUTPUT, "Unknown integration direction")


def integrate_chunk(chunk_args):
    """
    Integrates a chunk of tracers through 2D FLASH simulation snapshots.

    This function:
    1. Loads snapshots into shared memory for efficient parallel access.
    2. Integrates each active tracer in parallel using multiprocessing.
    3. Cleans up shared memory when done.

    Args:
        chunk_args (tuple): Contains all required arguments for this chunk:
            - plt_files_chunk: list of snapshot file paths
            - startpos: initial positions of tracers
            - tracer_entries: array storing tracer data
            - solve_ivp_args: arguments for solver
            - still_calc: boolean array marking which tracers are active
            - reached_NSE: boolean array for NSE tracers
            - still_check_NSE: boolean array for NSE checking
            - output_dir: path to write logs
            - tracer_entries_fmt: format string for tracer entries
            - time_limit: max allowed time for integration
            - td_vars_keys: keys of thermodynamic variables to extract
            - num_cpus: number of parallel processes to spawn
    Returns:
        np.ndarray: New positions of all tracers after integration
    """

    # --- Unpack arguments ---
    (plt_files_chunk, startpos, tracer_entries, solve_ivp_args,
     still_calc, reached_NSE, still_check_NSE, output_dir,
     tracer_entries_fmt, time_limit, td_vars_keys, num_cpus) = chunk_args

    # For storing snapshot metadata and shared memory handles
    snapshots_meta = []
    times_chunk = []
    shm_handles_all = []

    # --- Helper 1: Load snapshots into shared memory ---
    def load_snapshots_into_shm(file_list, td_vars_keys):
        """
        Loads snapshots and their thermodynamic variables into shared memory
        for efficient access by multiple processes.
        """
        meta_list = []
        time_list = []
        shm_list = []

        write_log(output_dir, "Reading in snapshots into shared memory...")
        t0 = time.time()

        for plt_path in file_list:
            snap = Snap.Snapshot2D(plt_path, td_vars_keys)
            time_list.append(snap.currentSimTime())

            # Thermodynamic variables in shared memory
            td_vars_meta, td_shms = create_shm_for_snapshot(snap, td_vars_keys)
            shm_list.extend(td_shms)

            # Other important arrays in shared memory
            meta_bbox, shm_bbox = create_shm_for_array("bbox_", snap.bbox)
            meta_rs, shm_rs = create_shm_for_array("realscalars_", snap.realscalars)
            meta_rrp, shm_rrp = create_shm_for_array("realruntimeparams_", snap.realruntimeparams)
            shm_list.extend([shm_bbox, shm_rs, shm_rrp])

            snap_meta = {
                "bbox": meta_bbox,
                "realscalars": meta_rs,
                "realruntimeparams": meta_rrp,
                "cells_per_block_x": snap.cells_per_block_x,
                "cells_per_block_y": snap.cells_per_block_y,
                "td_vars": td_vars_meta,
            }
            meta_list.append(snap_meta)

        write_log(output_dir, f"Snapshots loaded in {time.time() - t0:.2f}s")
        # Log current memory usage
        process = psutil.Process(os.getpid())
        write_log(output_dir, f"Current memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
        
        return meta_list, time_list, shm_list

    # --- Helper 2: Integrate tracers in parallel ---
    def integrate_tracers_parallel():
        """
        Integrates all active tracers in parallel using multiprocessing.
        Returns new positions and out-of-bounds events.
        """
        active_tracer_ids = np.where(still_calc)[0]
        new_positions = np.full((len(startpos), 2), np.nan)  # Initialize with NaN
        oob_list = []

        # Spawn pool for parallel computation
        with mp.Pool(processes=num_cpus) as pool:
            results = pool.starmap(
                integrate_single_tracer,
                [
                    (
                        tr_id,
                        startpos[tr_id],
                        times_chunk,
                        snapshots_meta,
                        sgn,       # If defined globally or in outer scope
                        keys,      # If defined globally or in outer scope
                        tracer_entries,
                        solve_ivp_args,
                        still_calc,
                        reached_NSE,
                        still_check_NSE,
                        output_dir,
                        tracer_entries_fmt,
                        time_limit
                    )
                    for tr_id in active_tracer_ids
                ]
            )

        # Merge results back into full array
        for idx, tr_id in enumerate(active_tracer_ids):
            new_positions[tr_id] = results[idx][0]
        for events in [res[1] for res in results]:
            oob_list.extend(events)

        return new_positions, oob_list

    # --- Helper 3: Clean up shared memory --

    try:
        # 1️⃣ Load snapshots into shared memory
        snapshots_meta, times_chunk, shm_handles_all = load_snapshots_into_shm(plt_files_chunk, td_vars_keys)

        # 2️⃣ Integrate active tracers
        t1 = time.time()
        new_positions, oob_events = integrate_tracers_parallel()
        write_log(output_dir, f"Integrated chunk in {time.time() - t1:.2f}s")

        # 3️⃣ Return final positions
        return np.array(new_positions)

    except Exception as e:
        print("❌ Exception in integrate_chunk:", e)
        traceback.print_exc()
        raise

    finally:
        # 4️⃣ Always clean up shared memory
        cleanup_shared_memory(shm_handles_all)


# ==============================================================================
#  TRACER INTEGRATION  (single tracer)
# ==============================================================================


def integrate_single_tracer(tr_id, start_pos, times_chunk, snapshots_meta, sgn, keys,
                            tracer_entries, solve_ivp_args, still_calc, reached_NSE,
                            still_check_NSE, output_dir, tracer_entries_fmt, time_limit):
    """
    Integrate a single tracer through the FLASH 2D velocity field.
    
    Uses shared-memory snapshots → no file I/O inside worker.
    Returns the tracer's final position and any out-of-bounds (OOB) events.
    """

    local_oob_events = []

    tracer_start_time = time.time()

    # --- Helper 1: Reconstruct snapshots from shared memory ---
    def reconstruct_snapshots(meta_list):
        """Rebuild Snapshot2D objects from shared memory metadata."""
        return [Snap.Snapshot2D.from_shm(snap_meta) for snap_meta in meta_list]

    # --- Helper 2: Velocity field function for solver ---
    def velocity_field(t, pos):
        """
        Interpolates the velocity field to position (x, y) and time t.
        Includes:
        - Linear interpolation between snapshots
        - Time-out protection
        - Forward/backward integration handling
        """
        x, y = pos

        # Wall-clock timeout check
        if (time.time() - tracer_start_time) > time_limit:
            msg = f'Tracer {tr_id}: timeout ({time_limit:.1f}s) at t={t:.5f}s, pos=({x/1e5:.6e},{y/1e5:.6e}) km'
            write_log(output_dir, msg)
            raise RuntimeError(msg)

        # Determine snapshot indices for interpolation
        if sgn == 1:  # forward integration
            if t <= times_chunk[0]:
                idx = 0
            elif t >= times_chunk[-1]:
                idx = -1
            else:
                idx_r = np.searchsorted(times_chunk, t, side='right')
                idx_l = idx_r - 1
                t_pair = [times_chunk[idx_l], times_chunk[idx_r]]
                velx_pair = [snapshots[idx_l].getQuantAtPos('velx', x, y, tr_id),
                             snapshots[idx_r].getQuantAtPos('velx', x, y, tr_id)]
                vely_pair = [snapshots[idx_l].getQuantAtPos('vely', x, y, tr_id),
                             snapshots[idx_r].getQuantAtPos('vely', x, y, tr_id)]
                return [np.interp(t, t_pair, velx_pair), np.interp(t, t_pair, vely_pair)]
        else:  # backward integration
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

        # Default: return velocity at nearest snapshot
        return [snapshots[idx].getQuantAtPos('velx', x, y, tr_id),
                snapshots[idx].getQuantAtPos('vely', x, y, tr_id)]

    # --- Helper 3: Out-of-bounds event function for solver ---
    def oob_event(t, pos):
        x, y = pos
        # returns negative if outside simulation domain
        return min(x, XMAX-x, y-YMIN, YMAX-y)

    oob_event.terminal = True
    oob_event.direction = -1  # triggers when tracer leaves domain

    # --- Helper 4: Sample variables at nearest snapshots ---
    def sample_vars_nearest(times, positions, keys, snapshots, times_chunk):
        sampled = {key: np.empty(len(times)) for key in keys}
        for i, (t, x, y) in enumerate(zip(times, positions[0], positions[1])):
            idx = np.abs(times_chunk - t).argmin()
            snap = snapshots[idx]
            for key in keys:
                sampled[key][i] = snap.getQuantAtPos(key, x, y, tr_id)
        return sampled

    try:
        # 1️⃣ Reconstruct snapshots
        snapshots = reconstruct_snapshots(snapshots_meta)

        # 2️⃣ Integration span and initial position
        t_span = [times_chunk[0], times_chunk[-1]]
        start_pos = [start_pos[0], start_pos[1]]

        # 3️⃣ Solve tracer trajectory
        result = solve_ivp(
            velocity_field,
            t_span=t_span,
            y0=start_pos,
            rtol=solve_ivp_args[0],
            atol=solve_ivp_args[1],
            max_step=solve_ivp_args[2],
            method='RK45',
            t_eval=times_chunk,
            first_step=min(1e-6, abs(t_span[1]-t_span[0])),
            events=oob_event
        )

        # 4️⃣ Handle out-of-bounds events
        if result.t_events[0].size > 0:
            still_calc[tr_id] = False
            exit_time = result.t_events[0][0]
            exit_pos = result.y_events[0][0]
            log_pos = (f"{np.round(exit_pos[0]/1e5,2)}, {np.round(exit_pos[1]/1e5,2)}"
                       if exit_pos is not None else "unknown")
            write_log(output_dir, f"Tracer {tr_id} OOB at t={exit_time:.3f}, (x,y)=({log_pos}) km")
            local_oob_events.append(f"Tracer {tr_id} OOB at t={np.round(exit_time,3)}, pos=({log_pos}) km")

            # Sample in-bounds variables
            vars_sampled = sample_vars_nearest(result.t, result.y, keys, snapshots, times_chunk)
            write_to_tracer_file(result.t, result.y, keys, vars_sampled, tracer_entries, tr_id, output_dir, tracer_entries_fmt)
            return np.array([result.y[0][-1], result.y[1][-1]]), local_oob_events

        # 5️⃣ Sample variables for full trajectory
        vars_sampled = sample_vars_nearest(result.t, result.y, keys, snapshots, times_chunk)

        # 6️⃣ Handle NSE conditions for backward tracers
        if sgn == -1 and still_check_NSE and 'temp' in vars_sampled:
            if max(vars_sampled['temp']) > NSE_TEMP:
                reached_NSE[tr_id] = True
                still_check_NSE[tr_id] = False
                if ONLY_UNTIL_MAXTEMP and max(vars_sampled['temp']) > MAXTEMP_TRACER:
                    still_calc[tr_id] = False
                    write_log(output_dir, f"Tracer {tr_id} reached max_temp and will be terminated")

        # 7️⃣ Write tracer data to file
        write_to_tracer_file(result.t, result.y, keys, vars_sampled, tracer_entries, tr_id, output_dir, tracer_entries_fmt)

        return np.array([result.y[0][-1], result.y[1][-1]]), local_oob_events

    except Exception as e:
        tb_str = traceback.format_exc()
        write_log(output_dir, f"❌ Tracer {tr_id} failed: {e}, tb: {tb_str}")
        still_calc[tr_id] = False
        return np.array([np.nan, np.nan]), [f"❌ Tracer {tr_id} failed: {e}"]

    finally:
        # 8️⃣ Always clean up shared memory
        for snap in snapshots:
            snap.close_shm()
