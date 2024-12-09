#!/bin/bash
#SBATCH --job-name=john
#SBATCH --output=optim.out
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=7

echo "========= Job started  at `date` on `hostname -s` =========="

#export I_MPI_HYDRA_TOPOLIB=ipl
#export OMP_NUM_THREADS=1

echo "Job id       : $SLURM_JOB_ID"


time mpiexec.hydra -n $SLURM_NTASKS julia fit_plot.jl


echo "========= Job finished at `date` =========="
