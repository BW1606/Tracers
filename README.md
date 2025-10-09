# 2-D FLASH Tracer Integration

Author: Benedikt Weinhold (~Oct 2025)  
Description: Python pipeline for placing and integrating Lagrangian tracer particles through 2D axisymmetric FLASH supernova simulations.  

## Disclaimer
  -  This is work in progress until i finish my master thesis end of 2026
---

## Aims
  - This  Repository includes a Pipeline to integrate the path of *Lagrangian Tracer Particles* throughout a astrophysical simulation from the output snapshots,
      all while saving the thermodynamic conditions ('temp', 'dens', 'ye', ...) in order to do a full post-processing nucleosynthesis analysis with a nuclear reaction network.
  - The format of the output files is made as an input for the open source nuclear reaction network WinNet (https://github.com/nuc-astro/WinNet)
  - The code is developed and tested for tracers in core collapse supernovae, but can in principle be used for any hydro-output regardless of the simulated event
  - As is, the code can read  and process the snapshots/plotfiles from the Hydro-Code FLASH but without much work this can be adjusted for ANY simulation code
     (more on that in the section *Snapshot2D class*)
  - There are some jupyter notebooks with elementary nucleosynthesis analysis added, these expect the nucleosynthesis calculation to be done with WinNet.

---

## Features

- **Tracer Placement**  
  - Place tracers based on density and Ye deviations.  
  - Supports increasing tracer density in regions where `|Ye - 0.5|` is significant.  

- **Tracer Integration**  
  - Integrates tracers through the FLASH velocity field using shared-memory interpolators.  
  - Supports forward and backward time integration with boundary checks and event termination (OOB, NSE).  
  - Parallelized with Python multiprocessing.  

- **Output Handling**  
  - Writes tracer positions and thermodynamic histories to text files.  
  - Handles multiple chunks and large numbers of tracers efficiently.  

---

## Requirements

- Python 3.10+  
- NumPy  
- SciPy  
- Multiprocessing  
- HDF5 reading library (e.g., `h5py` if needed by `Snapshot2D`)  

---

## Repository Structure

Tracers/
├── main.py # Main execution script
├── config.py # User-configurable parameters
├── src/
│ ├── init.py
│ ├── utils.py # Utility functions (logging, etc.)
│ ├── tracer_files.py # File handling for tracer output
│ ├── tracer_placement.py # Functions for placing tracers in snapshots
│ ├── tracer_integration.py # Integration routines (solve_ivp-based)
│ ├── Snapshot2D.py # FLASH snapshot reading and handling
│ └── Progenitors.py # Progenitor-specific input handling
├── logs/ # Output logs
├── README.md # This file
└── basic_nuclear_analysis(soon)

---

## Usage

1. **Configure parameters** in `config.py`, including:
   - Snapshot directory
   - Number of tracers
   - Integration settings
   - Output directory  

2. **Run the program** via SLURM or locally:
```bash
python main.py
```
3. Monitor progress through log file run.log.

---
## List of parameters
  - In the future (TODO)

## Citation

No citation is required, but an acknowledgement is appreciated (Also because im interested in your results :))
