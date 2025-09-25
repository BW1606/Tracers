# CalcTracers.py

**Author:** Benedikt Weinhold – 10/09/2025

## Disclaimer
-until I finish my Master-Thesis in the december of 2025, this is still work in progress. The program is ready for use but there might be quality of life improvements in the future, Furthermore in the end i will add some analysis tools to calculate and visualize the composition of the calculated tracers and the simulation in order to physically understand why, what kind of composition resulted in the simulation. These analysis tools will assume the composition of the tracers were calculated with the help of the nuclear reaction network WinNet (https://github.com/nuc-astro/WinNet)

---

## Dependencies
- Python (Anaconda recommended)
- `Snapshot2dFLASH.py` class, or an equivalent class for a different hydrodynamics code output  
  *(see “Hydro Snapshot class” section for details)*
- `Progenitors.py` with a working progenitor class for the used progenitor file

---

## Aims
This program integrates the paths of Lagrangian tracer particles throughout a hydrodynamic astrophysical simulation using snapshot files, with the RKF-45 integration method implemented in `scipy.integrate.solve_ivp`.  

- Developed for **2D axisymmetric simulations**, but can be generalized to 3D with some adjustments.
- Supports **forward and backward integration** in time.  
  - Backward integration is recommended for nucleosynthesis analysis.
- Particle placement:  
  - Either **proportional to density**, or  
  - **From a file** specifying initial positions.
- If needed (e.g., for nucleosynthesis analysis), calculates **initial composition of tracers** from a progenitor file.

---

## How the code works

1. The code places the tracers according to the placement parameters (only_unbound, maxTemp, maxDens) and depending on the integration direction (backwards/forwards) at the first or last snapshot of the simulation.
2. Afterwards the amount of snapshots will be chunked and with one chunk at a time the code calls integrate_chunk(args). Here the snapshots will be read in as objects of the class Snapshot2dFLASH (or simmilar (more on that later)) and a multiprocessing pool will be initiated. Now each worker calls the integrate_single_tracer(args) funciton with the start position of the tracer at the time of the chunk. integrate_single_tracer(args) handles the integration of the tracer throughout the simulation domain and the times in the chunk, after the chunk the data will be saved in the tracerfiles set up in path_to_tracers.
3. integrate_single_tracer(args): gets various arguments (see in the code) but most imortantly the list of snapshot objects for the specific chunk and the starting position of the tracer. The heart of the function is the integration with scipy.integrate.solve_ivp which requires a function velocity_field(pos, t) which gives back the velocities in x- and y-direction at a given time and space. After integrating the path throughout the chunk all the thermodynamic data required/specified by the user (calcNeutrinos = True/False, or custom keys) are calculated at the timesteps of the snapshots (every 1ms in my simulations). This requires the class Snapshot2dFLASH to have a method calles snap.getQuantAtPos(self, pos, key) which gives back the value of key interpolated to the position pos (more on that later).... 

---

## Checklist before running

1. **Snapshot path**: Set `path_to_pltfiles` to the directory containing snapshots.  
2. **Tracer output path**: Set `path_to_tracers` (directory will be created automatically).  
   - **Note:** If the directory exists, the run is aborted to avoid overwriting previous data.  
3. **Initial composition**: Set `calc_seeds = True/False`.  
   - If `True`, specify:  
     - The progenitor file type (currently only MESA output and FLASH input format supported)  
     - The path to the progenitor file: `path_to_progfile`  
4. **Integration direction**: Forward or backward in time.  
   - Backward integration is recommended for nucleosynthesis.  
5. **Placement method**: `'PosWithDens'` or `'FromFile'`.  
   - `'PosWithDens'`:  
     - `num_tracers`: Approximate number of tracers to place  
     - `only_unbound = True/False`  
       - If `True`: Place tracers only in **unbound cells** (ener + gpot > 0 & v_rad > 0)  
       - If `False`: Specify `max_dens` (tracers are not placed in cells denser than this, e.g., PNS)  
     - **Note:** At least one tracer is placed per block that meets the criteria. This means the total number of tracers may exceed `num_tracers`.  
   - `'FromFile'`:  
     - Reads tracer starting positions from a file.  
     - WARNING: these tracers are assumed to have mass = 1g  
     - Specify the path to the file  

6. **Neutrino info**: Set `calc_neutrinos = True/False`  

---

## Running the code

- Using SLURM, the provided script can be run with:  
  ```bash
  sbatch runTracers.sh
- Alternatively, run locally with (might want to limit the amount of CPU's)
  ```bash
  python CalcTracers.py

---

## Monitor a running calcuation

- The run.log file has information about the progress of the code - try it out:  
  ```bash
  tail -f /path/to/output_dir/run.log
  
 
