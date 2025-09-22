from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

class Progenitor(ABC):
    """
    Abstract base class for progenitor models.
    
    Attributes
    ----------
    radius : np.ndarray
        Radial grid of the progenitor [cm].
    abundances : Dict[str, np.ndarray]
        Mass fractions of nuclear species at each radius.
    """

    # species metadata: Z = proton number, A = mass number
    nuclear_species_meta = {
        'neut': {'Z': 0,  'A': 1},
        'h1':   {'Z': 1,  'A': 1},
        'he3':  {'Z': 2,  'A': 3},
        'he4':  {'Z': 2,  'A': 4},
        'c12':  {'Z': 6,  'A': 12},
        'n14':  {'Z': 7,  'A': 14},
        'o16':  {'Z': 8,  'A': 16},
        'ne20': {'Z': 10, 'A': 20},
        'mg24': {'Z': 12, 'A': 24},
        'si28': {'Z': 14, 'A': 28},
        's32':  {'Z': 16, 'A': 32},
        'ar36': {'Z': 18, 'A': 36},
        'ca40': {'Z': 20, 'A': 40},
        'ti44': {'Z': 22, 'A': 44},
        'cr48': {'Z': 24, 'A': 48},
        'cr56': {'Z': 24, 'A': 56},
        'fe52': {'Z': 26, 'A': 52},
        'fe54': {'Z': 26, 'A': 54},
        'fe56': {'Z': 26, 'A': 56},
        'ni56': {'Z': 28, 'A': 56}
    }

    def __init__(self, radius: np.ndarray, abundances: Dict[str, np.ndarray]):
        self.radius = np.asarray(radius)
        self.abundances = abundances

    @abstractmethod
    def massfractions_of_r(self, r: float) -> Dict[str, Dict]:
        """
        Return mass fractions of all nuclear species at radius r including metadata.
        Each species entry is a dictionary: {'Z':..., 'A':..., 'X':...}
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}: {len(self.radius)} zones, species={list(self.abundances.keys())}>"


class Progenitor_NuGrid(Progenitor):
    """
    Progenitor data coming from NuGrid.
    """

    def __init__(self, path_to_progfile):
        prog_dat = pd.read_table(path_to_progfile, sep='\s+', skiprows=5)
        R_sun_cm = 6.9598e10
        radii = prog_dat['radius'].values * R_sun_cm

        # read abundances (X) only
        species_cols = list(prog_dat.columns[63:84]) #this is specific to NuGrid prog file
        abundances = {col: prog_dat[col].values for col in species_cols}

        super().__init__(radii, abundances)

    def massfractions_of_r(self, r: float) -> Dict[str, Dict]:
        results = {}

        # sum h1 + prot if present
        h1_sum = 0.0
        for sp in ['h1', 'prot']:
            if sp in self.abundances:
                f = interp1d(self.radius, self.abundances[sp], bounds_error=False, fill_value=0.0)
                h1_sum += float(f(r))
        if h1_sum > 0.0:
            meta = self.nuclear_species_meta.get('h1', {'Z': 1, 'A': 1})
            results['h1'] = {'Z': meta['Z'], 'A': meta['A'], 'X': h1_sum}

        # now handle all other species
        for sp, data_array in self.abundances.items():
            if sp in ['h1', 'prot']:
                continue
            f = interp1d(self.radius, data_array, bounds_error=False, fill_value=0.0)
            X_val = float(f(r))
            meta = self.nuclear_species_meta.get(sp, {'Z': 0, 'A': 0})
            results[sp] = {'Z': meta['Z'], 'A': meta['A'], 'X': X_val}

        return results


class Progenitor_FLASH(Progenitor):
    """
    Progenitor data coming from a progenitor dat file made as FLASH input

    """


    def __init__(self, path_to_progfile):
        with open(path_to_progfile, "r") as f:
            header_lines = [line.strip() for line in f.readlines()[:30]]

        column_names = ['radius']


        for line in header_lines[2:]:
            column_names.append(line)

        prog_dat = pd.read_table(path_to_progfile, sep=r"\s+", skiprows=30, header=None)
        prog_dat.columns = column_names

        species_cols = list(prog_dat.columns[9:30])
        abundances = {col: prog_dat[col].values for col in species_cols}

        super().__init__(prog_dat['radius'], abundances)

    
    def massfractions_of_r(self, r: float) -> Dict[str, Dict]:
        results = {}

        for sp, data_array in self.abundances.items():
            f = interp1d(self.radius, data_array, bounds_error=False, fill_value=0.0)
            X_val = float(f(r))
            meta = self.nuclear_species_meta.get(sp, {'Z': 0, 'A': 0})
            results[sp] = {'Z': meta['Z'], 'A': meta['A'], 'X': X_val}

        return results


