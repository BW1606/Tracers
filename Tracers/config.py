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
# PATH_TO_PLTFILES = ('/home/template_scripts/FLASH/'             # where and how are the simulation outputs saved/names
#                     '2D_simulation/HeS_s13.8/HeS_hot/'
#                     'HeS_hot_hdf5_plt_cnt_')   
# PATH_TO_OUTPUT = ('/home/bweinhold/Auswertung/2D_Analysis/'     # output-directory for tracers, seeds and run.log ()
#                    '2D_Tracers/data/Simulations/HeS_s13.8/'     # will be created, if already exist - code aborts
#                    '10ktr_bwd_pwd_oub_wS_wNu_yecrit_800ms_shm/')
# PATH_TO_PROGFILE = ('/home/template_scripts/FLASH/'             # path to progenitorfile
#                     '2D_simulation/HeS_s13.8/HeS_s13.8.1d')  

PATH_TO_PLTFILES = '/home/template_scripts/FLASH/2D_simulation/S15_Ritter_hf10_2d/output/S15_Ritter_hf10_hdf5_plt_cnt'
PATH_TO_OUTPUT = '/home/bweinhold/Auswertung/2D_Analysis/2D_Tracers/data/Simulations/S15_Ritter_hf10/test_pos_2_shm'
PATH_TO_PROGFILE = '/home/bweinhold/Auswertung/2D_Analysis/2D_Tracers/data/Progenitors/S15_NuGrid_log423.data'

# Get paths for Snapshots
PLT_FILES = sorted(glob(PATH_TO_PLTFILES + "*", recursive=False))#[750:801]

# Arbitrary Message to run.log
ARB_MESSAGE = 'Testing modularized code'

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
PLACEMENT_METHOD = 'FromFile'                                # 'PosWithDens' or 'FromFile', see README
NUM_TRACERS = 10000                                             # if PosWithDens: number of tracers to place (duh?!)                
ONLY_UNBOUND = True                                             # if PosWithDens: only place tracers in ejected areas
MAX_TEMP_PLACE = 1e10                                           # if backwards: Dont place tracers above 
MAX_DENS = 1e11                                                 # if not ONLY_UNBOUND: dont place tracers in areas with dens > MAX_DENS (i.e. in PNS)
PATH_TO_TRACERS_START = '/home/bweinhold/Auswertung/2D_Analysis/2D_Tracers/data/test_positions/test_positions_bw_2.dat'                                      # if FromFile: where is the file with the positions


# Progenitor                                                    
CALC_SEEDS = True                                               # Calculates initial composition of tracer from progenitor file
PROG_TYPE = 'NuGrid'                                             # type of progenitor file must be in progenitors.py (see README)                   

# Domain bounds
XMIN, XMAX, YMIN, YMAX = 0, 3.2e9, -3.2e9, 3.2e9