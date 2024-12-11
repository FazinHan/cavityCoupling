#!/bin/bash
#SBATCH --job-name=john
#SBATCH --output=optim.out
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=7

time mpiexec.hydra -n $SLURM_NTASKS julia fit_plot.jl