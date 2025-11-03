import numpy as np
import pandas as pd
from glob import glob
import os
import h5py
import re


"""
Still TODO:
    - manage WinNet out with snapshot out

"""

class WinNet_output:

    def __init__(self, output_dir, with_NSE_flag = False, 
                want_tracer_info = False, want_mainout_info = False,
                want_finabs_info = False, has_snapshots = False, want_all = False):
        
        self.output_dir = output_dir

        if want_all:
            want_finabs_info=True
            want_mainout_info=True
            want_tracer_info=True
                
        if want_tracer_info:
            self.path_to_tr = str(sorted(glob(os.path.join(output_dir, "tracer*.dat")))[0])
            match = re.search(r"tracer(\d{5})\.dat", self.path_to_tr)
            if match:
                number_str = match.group(1)
                self.tracer_id = int(number_str)

            if with_NSE_flag:
                self.tracer_data, self.tracer_data_units, self.mass, self.mass_unit, self.reached_NSE = self._read_tracer_file(self.path_to_tr, with_NSE_flag)
            else:
                self.tracer_data, self.tracer_data_units, self.mass, self.mass_unit = self._read_tracer_file(self.path_to_tr, with_NSE_flag)
        
        if want_mainout_info:
            self.path_to_mo = str(sorted(glob(os.path.join(output_dir, "mainout*")))[0])
            self.mainout_data, self.mainout_data_units = self.read_mainout_file(self.path_to_mo)
        
        if want_finabs_info:
            self.path_to_wn = str(sorted(glob(os.path.join(output_dir, "WinNet_data*.h5")))[0])
            self.path_to_seed = str(sorted(
                glob(os.path.join(output_dir, "seed*.dat")) +
                glob(os.path.join(output_dir, "seed*.txt"))
            )[0])

            if has_snapshots:
                self.finabsum, self.finabelem = self.read_WinNet_data(self.path_to_wn, has_snapshots)
            else:
                self.finabsum, self.finabelem = self.read_WinNet_data(self.path_to_wn)
        
        if not want_tracer_info and not want_mainout_info and not want_finabs_info and not want_all:
            print('Choose the type of info u need: want_tracer_info, want_mainout_info or want_finabs_info or want_all with True/False')


    def _read_tracer_file(self, file_path, NSE_flag_bool):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        mass_line = lines[0].strip()

        if NSE_flag_bool:
            pattern = r"Mass of the tracer \[([^\]]+)\]: ([\deE\+\.-]+), reached_NSE: (True|False)"
        else:
            pattern = r"Mass of the tracer \[([^\]]+)\]: ([\deE\+\.-]+)"
        match = re.search(pattern, mass_line)

        if not match:
            raise ValueError(f"Could not parse tracer header in {file_path}")

        mass_unit = match.group(1)          # text inside [ ]
        mass = float(match.group(2))        # numeric mass value
        if NSE_flag_bool:
            reached_NSE = (match.group(3) == "True") 

        # --- Extract header (second line) ---
        header_line = lines[1].lstrip("# ").rstrip("\n")  # remove leading '#' and trailing newline

        # Regex: match column name (\S+) optionally followed by [unit]
        pattern = r'(\S+)\s*(?:\[(.*?)\])?'
        matches = re.findall(pattern, header_line)

        col_names = [m[0] for m in matches]
        col_units = [m[1] for m in matches]

        # --- Read data skipping comment lines ---
        data = np.loadtxt(file_path, comments='#')

        # --- Safety check ---
        n_cols = min(len(col_names), data.shape[1])
        data_dict = {name: data[:, i] for i, name in enumerate(col_names[:n_cols])}
        unit_dict = {name: unit for name, unit in zip(col_names[:n_cols], col_units[:n_cols])}

        if NSE_flag_bool:
            return data_dict, unit_dict, mass, mass_unit, reached_NSE
        else:
            return data_dict, unit_dict, mass, mass_unit

    def read_mass_fraction_file(self, file_path):
        """
        Reads a file like:
        # A Z X
        # ---------------
        1 1 3.891945e-08
        1 0 8.782083e-09
        ...
        
        Returns a dictionary:
        { 'A': np.array([...]), 'Z': np.array([...]), 'X': np.array([...]) }
        """

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Find the header line (first line starting with '#')
        for line in lines:
            if line.startswith("#") and not set(line.strip()) <= {"#", "-"}:  # skip separator lines
                header_line = line.lstrip("# ").strip()
                break
        else:
            raise ValueError("No header line found in file")

        # Column names
        col_names = header_line.split()  # simple split, e.g. ['A', 'Z', 'X']

        # Read data, skipping all comment lines
        data_lines = [line for line in lines if not line.startswith("#")]
        data = np.array([list(map(float, line.split())) for line in data_lines])

        # Build dictionary
        n_cols = min(len(col_names), data.shape[1])
        data_dict = {name: data[:, i] for i, name in enumerate(col_names[:n_cols])}

        return data_dict
    

    def read_mainout_file(self, file_path):
        """
        Reads mainout.dat-like files with multi-line headers containing units in [].
        Returns:
        data_dict : {column_name: numpy array of values}
        unit_dict : {column_name: unit string from [] or empty string}
        """

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # --- Extract header lines ---
        header_lines = []
        for line in lines:
            if line.startswith("#"):
                header_lines.append(line.lstrip("# ").strip())
            else:
                break

        # Concatenate header lines and remove trailing commas
        header_str = " ".join(header_lines).replace(",", "")

        # Remove numbering prefixes like "1:", "2:" etc.
        header_str = re.sub(r'\d+:\s*', "", header_str)

        # --- Extract names and units ---
        col_names = []
        col_units = []

        # Split header into tokens
        tokens = header_str.split()
        for tok in tokens:
            # Match name[unit] or name
            m = re.match(r'([^\[\]]+)\[([^\]]+)\]', tok)
            if m:
                name = m.group(1)
                unit = m.group(2)
            else:
                name = tok
                unit = ""
            col_names.append(name)
            col_units.append(unit)

        # --- Read numeric data ---
        data = np.loadtxt(file_path, comments='#')

        # Safety: in case header has more names than columns
        n_cols = min(len(col_names), data.shape[1])
        data_dict = {name: data[:, i] for i, name in enumerate(col_names[:n_cols])}
        unit_dict = {name: unit for name, unit in zip(col_names[:n_cols], col_units[:n_cols])}

        return data_dict, unit_dict
    

    def read_WinNet_data(self, file_path, has_snapshots = False):

        """
        Reads the WinNet HDF5 output file (WinNet_data*.h5) and extracts final abundances.

        The file contains two main groups:
        - 'finab/finabsum'  : summed abundances of nuclei per massnumber
        - 'finab/finabelem' : individual element abundances

        Returns:
        ----------
        finabsum : dict
            Dictionary containing arrays of summed abundances with keys:
            'Y', 'X', 'A' (mass fraction, mass number, etc.)
        finabelem : dict
            Dictionary containing arrays of individual element abundances with keys:
            'Y', 'X', 'Z' (mass fraction, mass number, charge number, etc.)
        """
        finabsum = {}
        finabelem = {}

        snapshots_A = []
        snapshots_Z = []
        
        if has_snapshots:
            with h5py.File(file_path, "r") as h5_file:
                finabelem.update({'Y' : h5_file['finab/finabelem/Y'][:], 'X' : h5_file['finab/finabelem/X'][:], 'Z' : h5_file['finab/finabelem/Z'][:]})
                finabsum.update({'Y' : h5_file['finab/finabsum/Y'][:], 'X' : h5_file['finab/finabsum/X'][:], 'A' : h5_file['finab/finabsum/A'][:]})
                
                #now read in snapshot abundances
                snapshot_times = h5_file['snapshots/time'][:]
                all_A = h5_file['snapshots/A'][:]
                all_Z = h5_file['snapshots/Z'][:]
                all_Y = h5_file['snapshots/Y'][:]
                
            #If one wants isotopic information this is the raw data to use
            #Here we only extract abundances or massfractions of A and Z

            #first create abundances and mass fractions of A
            for t in range(len(snapshot_times)):
                all_Y_snap = all_Y[t]
                A_snap = np.arange(1, all_A.max()+1)
                Y_snap = np.bincount(all_A, weights=all_Y_snap)[A_snap]
                snapshots_A.append({'t' : snapshot_times[t], 'A': A_snap, 'Y' : Y_snap, 'X': Y_snap*A_snap})

                Z_snap = np.arange(1, all_Z.max()+1)
                Y_snap_elem = np.bincount(all_Z, weights=all_Y_snap)[Z_snap]
                snapshots_Z.append({'t' : snapshot_times[t], 'Z': Z_snap, 'Y' : Y_snap_elem, 'X': Y_snap_elem*Z_snap})


            self.snapshot_times = snapshot_times
            self.snapshot_composition = snapshots_A
            self.snapshot_composition_elem = snapshots_Z
                    

        else:
            with h5py.File(file_path, "r") as h5_file:
                finabelem.update({'Y' : h5_file['finab/finabelem/Y'][:], 'X' : h5_file['finab/finabelem/X'][:], 'Z' : h5_file['finab/finabelem/Z'][:]})
                finabsum.update({'Y' : h5_file['finab/finabsum/Y'][:], 'X' : h5_file['finab/finabsum/X'][:], 'A' : h5_file['finab/finabsum/A'][:]})


        return finabsum, finabelem


# test_obj = WinNet_output('/home/bweinhold/Auswertung/2D_Analysis/nucleosynthesis_analysis/HeS_s13.8/10k_nf1_7GK_800ms_snaps/00009',
#                          with_NSE_flag=True, want_finabs_info=True, has_snapshots=True, want_tracer_info=True)

# snapshot_info = test_obj.snapshot_compositon

# print(test_obj.snapshot_times)
# print(test_obj.tracer_id)

# t1 = snapshot_info[1]['t']
# A_1 = snapshot_info[1]['A']
# X_1 = snapshot_info[1]['X']

# print(t1)

# print(A_1)
# print(X_1)
