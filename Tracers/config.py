# ==============================================================================
#  2-D FLASH TRACER INTEGRATION
#  Author: Benedikt Weinhold  (~Oct 2025)
#
#  CONFIG
#
#  Choose wanted parameters here, then execute main.py
#  For further explanations on the parameters see README.md 
# ==============================================================================

import numpy as np
from glob import glob

# Paths
PATH_TO_PLTFILES = ('/path/to/simulation/snapshots')   
PATH_TO_OUTPUT = ('/path/to/output/dir')
PATH_TO_PROGFILE = ('/path/to/progenitor/file')     #more on that in README (at some point)  

# Get paths for Snapshots
PLT_FILES = sorted(glob(PATH_TO_PLTFILES + "*", recursive=False))#[750:801]

# Arbitrary Message to run.log
ARB_MESSAGE = 'Arbitrary user message that will be printed in run.log'

# Integration
DIRECTION = 'backward'                                          # 'forward' or 'backward' - time direction of integration
CHUNK_SIZE = 100                                                # tracers are integrated throughout the simulation in chunks
RTOL, ATOL, MAXSTEP = 1e-2, 1e4, 1e-4                           # tolerances for solve_ivp/RKF5 - see README
TIME_LIMIT = 60.0                                               # single tracer time limit to avoid convergance problems

# Physics
WITH_NEUTRINOS = True                                           # also stores luminosity and mean energy per neutrino in tracerfiles
ONLY_UNTIL_MAXTEMP = False                                      # if backwards: calculates tracer only until MAXTEMP_TRACER
MAXTEMP_TRACER = 1e10 #[K]                                      # temps in K
NSE_TEMP = 5.8e9 #[K]                                           # if tracer reaches NSE its written into the header

# Tracer placement
PLACEMENT_METHOD = 'FromFile'                                   # 'PosWithDens' or 'FromFile', see README
NUM_TRACERS = 10000                                             # if PosWithDens: number of tracers to place (duh?!)                
ONLY_UNBOUND = True                                             # if PosWithDens: only place tracers in ejected areas
MAX_TEMP_PLACE = 1e10                                           # if backwards: Dont place tracers above 
MAX_DENS = 1e11                                                 # if not ONLY_UNBOUND: dont place tracers in areas with dens > MAX_DENS (i.e. in PNS)
PATH_TO_TRACERS_START = '/path/to/starting/pos'                 # if FromFile: where is the file with the positions


# Progenitor                                                    
CALC_SEEDS = True                                               # Calculates initial composition of tracer from progenitor file
PROG_TYPE = 'NuGrid'                                            # type of progenitor file must be in progenitors.py (see README)                   

# Domain bounds
XMIN, XMAX, YMIN, YMAX = 0, 3.2e9, -3.2e9, 3.2e9                # simulation boundaries 
