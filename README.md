# CalcTracers.py

**Author:** Benedikt Weinhold – 10/09/2025

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

1. Will be added later

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
  
 
