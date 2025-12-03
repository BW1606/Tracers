#!/bin/bash
#SBATCH --job-name=2D_Tracers
#SBATCH --output=logs/Tracers_2D_%j.out
#SBATCH --error=logs/Tracers_2D_%j.err
#SBATCH --partition=ComputeNew
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
##SBATCH --time=3:00:00                           

# Run your Python script with N processes internally

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun python main.py