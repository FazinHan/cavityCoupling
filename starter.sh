#!/bin/bash
#SBATCH --job-name=g1.g2
#SBATCH --output=optim.out
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=7
#SBATCH -A physics_engg

source ~/.bashrc

time mpiexec -n $SLURM_NTASKS julia fit_plot.jl
