This is the first rough version of the README for the CalcTracers.py file

@author Benedikt Weinhold - 10/09/2025

Dependencies:
    - python anaconda things
    - Snapshot2dFLASH.py class or analogue for a different Hydro Code output (More details somewhere else)
    - Progenitors.py with a working progenitor class for the used progenitor file
    - ensure_ascending_times.py

Aims:
    - this program integrates the path of langrangian tracer particles throughout a hydrodynamic astrophysical simulation using snapshot files using the RKF-45 integration method implemented in scipy.integrate.solve_ivp (made/tested for a CCSN simulation - should work for other, maybe need other tolerances and tracer placement methods)
        -this has been developed for 2D axisymmetric simulations but with some technical adjustments this can be generalized for 3D simulations aswell
    - supports forward and backward integration in time (backwards is recommended for nucleosynthesis analysis, more on that in my thesis/various papers on the topic)
    - Either places particles in the simulation domain proportional to density or takes starting positions from a file (more on that somewhere else)
    - If needed (i.e. for nucleosynthesis analysis): Calculates the initial composition of the tracers from the progenitor file

Checklist before run:
    - declared where the snapshots are in path_to_pltfiles = 
    - declared where the tracerfiles should be saved (the program creates this directory automatically) in path_to_tracers
        -if the directory already exists the run will be aborted to avoid overiding/mixing data of different runs :)
    - declared if initial composition of the tracers is wanted with calc_seeds = True/False
        - if True:
            - declared what kind of progenitor file (until now only NuGrid supported, more in the future)
            - declared where the progenitor file is in path_to_progfile
    - declared what time direction (backwards/forward) - pro/cons/discussion in moritz paper and my master thesis (an refs therein)
    - declared what placement method is wanted ('PosWithDens' or 'FromFile')
        -if 'PosWithDens': (Places tracers proportional to density throughout the simulation domain, more info in my thesis/somewhere)
            - declared num_tracers, the approximate amount of tracers that you want
            - declared if only_unbound = True/False (if true, place tracers only in unbound cells at the end of the simulation)
                -if only_unbound=False: specified a max_dens (e.g. 1e11g/cm**3) above which no tracer should be placed (no tracers in PNS)
            - Note: there will be at least one tracer per block that contains a cell with the chosen criterion (only_unbound or maxDens). even if num_tracer=100 there might me more that 1000 tracers placed according to how many blocks have at least one unbound/ejected cell
        
        - if 'FromFile': (reads in tracer starting positions from an external file. WARNING: these tracers cannot have mass/ are taken to have mass = 1g)
            - specified the path to the file in which the positions are, an example file for the format can be found in ... (still TODO)
    - declared if the run should be calculating and saving neutrino information with calc_neutrinos = True/False

Running the Code:
    - the slurm script used by me will be provided and can be run with 'sbatch runTracers.sh' as a cmd-line
    - while the code is running, go to the tracer directory and monitor progress with detailed information with 'tail -f run.log'

How does the code work:
    -later

List of parameters:

More Details:
    i) Placement methods
        -Function: Places tracers in the simulation domain according to the given parameters:
            - num_tracers: Number of tracers to place (care see last bullet point)
            - only_unbound = True/False: 
                A) If true: only place tracers in cells which are unbound (ener + gpot > 0 & vrad > 0)
                B) If False: only place tracers in cells with dens < maxDens (e.g. dont place tracers in the Neutron Star)
            - for nucleosynthesis analysis it is recommended to use only_unbound=True since otherwise a lot of tracers will be calculated that are   never ejected and therefore must be removed before calculating the final abundances (If the user is integrating backwards in time, which is also recommended)
            - While the number of tracers in a given area is proportional to the mass in the area, and therefore all tracers have similar masses, it is hardcoded to place at least one tracer per block which has unbound cells. This is to better capture bubbles of material that has different composition as the surrounding matter.
                -this has as a consequence that the minimum amount of placed tracers is the amount of blocks in the simulation domain that has unbound cells at the time of placement. (for flash, ca. 1s post bounce this might be around 1500-2000 minimum placed tracers). This should however not be a problem since the amount of tracers one needs to accurately acces the final abundances of the matter, one should use more than this (for 2D, axisymmetric simulations 10000-20000 tracers seem optimal)

    ii) Hydro Snapshot class
        -the Snapshot class reads in every snapshot as an object and only saves the information that is important for the tracer calculation as attributes, the class in this depository is for 2D, axisymmetric FLASH simulations, but can be taken as a blueprint for others.
        
        - making a class for another simulation code:
            -the tracer integration itself only needs the Snapshot class to provide these following functions:
                A)  snap.getQuantAtPos('key', x, y) which takes the key of the the wanted quantity (velocities, temperatures, densities, nutrino-quantites, ...) and the position in x, y and returns the value of the key at this position. For precision it is vital that one does not just give the value of the next cell of the simulation output but interpolates to the exact precision x,y (for example see Snapshot2dFLASH.py for a code, where the area is divided in blocks and each block has NxN cells)

                B) snap.currentSimTime() which returns the simulation time the snapshot is taken (should be easily accessable in all types of simulation outputs)

                C) snap.Snapshot2D('path/to/sim_snapshot', keys), the constructor of the class. Takes the path to the simulation output file and keys, a list of names of the data from the snapshot file that one wants in the tracer files in the end. keys HAS to include 'velx', 'vely' for the integration method in integrate_single_tracer(args) and 'dens', 'ener', 'gpot', the density, total energy and gravitational potential at each cell if the user wants to use the implemented tracer placement method PosWithDens(). An example can be taken from the keys in the code itself for the cases with_neutrinos = True/False

    iii) Progenitor Class
        -class to read in progenitor files needed to calculate the initial composition in terms of massfractions for the tracers

        -must contain a method: progenitor.massfractions_of_r(tr_init_radius) that takes the radius r of the tracer at the start of the simulation (i.e. at which time the progenitor composition holds true) and gives back a dictionary {'key': 'A', 'Z', 'X'} i.e. {'h1': '1', '1', '0.1688888'}. A is the mass number, Z is the charge/proton number and X is the massfraction. For examples check the implemented progenitor classes is progenitor.py