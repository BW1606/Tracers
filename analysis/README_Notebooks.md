# README – `analyze_single.ipynb`

This notebook provides tools to analyze the nucleosynthesis history of **individual Lagrangian tracer particles** from core‑collapse supernova simulations. It is designed to work with the `WinNet_output` class and associated tracer data stored in the WinNet output directory.

## 📘 Purpose

The notebook allows you to:
- Load and inspect the output of single tracers.
- Compare network-derived quantities (e.g., Ye, composition) to tracer‑integrated values.
- Plot time evolution of thermodynamic variables.
- Visualize abundances at selected times or over full trajectories.
- Perform diagnostic checks on network/tracer consistency.

## 📁 Requirements

The notebook expects:
- A fully functional `WinNet_output` class.
- Tracer output files written by WinNet.
- Paths to simulation and output directories adjusted in the notebook.

## 🧩 Main Features

### 1. **Load Single Tracer Data**
Imports a tracer’s nuclear network output and stores:
- Temperature / density evolution  
- Radius, velocity, entropy  
- Abundances of all nuclei  
- Electron fraction \( Y_e \)

### 2. **Plot Composition**
Generates:
- Abundance vs. mass number plots  
- Comparisons of abundances at different time snapshots  
- Network vs. tracer-based Ye evolution  

### 3. **Trajectory Visualization**
Includes:
- Temperature–time, density–time curves  
- Radius and velocity evolution  
- Entropy history  

### 4. **Diagnostics**
Helps identify:
- Deviations between network Ye and hydrodynamic tracer Ye  
- Effects of interpolation and resolution  
- Potential inconsistencies in tracer thermodynamic histories  

## ▶️ Usage

1. Place `analyze_single.ipynb` in the same environment where the `WinNet_output` class is available.
2. Adjust any file paths in the first notebook cells.
3. Run the notebook top‑to‑bottom.
4. Use the final plotting utilities for analysis or integration into your thesis figures.

## 🧪 Notes

- The notebook is part of a larger analysis workflow (e.g., `analyze_all.ipynb`).
- Several plots are optimized for publication-quality output.
- For large tracer datasets, execution time may vary.

## 📄 Author
Benedikt Weinhold (~approx Oct. 2025)

# READNE - `analyze_all.ipynb`

Still toDO