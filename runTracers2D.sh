!/bin/bash
#SBATCH --job-name=2D_Tracers_HeS
#SBATCH --output=logs/Tracers_HeS_2D_%j.out
#SBATCH --error=logs/Tracers_HeS_2D_%j.err
#SBATCH --partition=Compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4096
##SBATCH --time=3:00:00           
##SBATCH --mem=32G
                

# Run your Python script with N processes internally
python CalcTracers_4.py
