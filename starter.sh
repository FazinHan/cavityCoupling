#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=g1.g2
#SBATCH --output=optim.out
#SBATCH --time=4-00:00:00
#SBATCH --ntasks-per-node=7
#SBATCH -A physics_engg
#SBATCH --mail-user=fizaan.khan.phy21@iitbhu.ac.in

time mpiexec -n $SLURM_NTASKS /scratch/fizaank.phy21.iitbhu/julia-1.11.2/bin/julia fit_plot.jl
