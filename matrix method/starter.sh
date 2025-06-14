#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=matrics
#SBATCH --output=output_%j.out
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH -A physics_engg
#SBATCH --mail-user=fizaan.khan.phy21@iitbhu.ac.in
#SBATCH --partition=cpu

WORKDIR="/scratch/fizaank.phy21.iitbhu/cavityCoupling/matrix method/jobs/$SLURM_JOB_ID"
mkdir -p "$WORKDIR" && cd "$WORKDIR" || exit -1

echo "========= Job started  at `date` on `hostname -s` =========="

#export I_MPI_HYDRA_TOPOLIB=ipl
#export OMP_NUM_THREADS=1

echo "Job id       : $SLURM_JOB_ID"

export I_MPI_FALLBACK=disable
export I_MPI_FABRICS=shm:tmi
export I_MPI_DEBUG=9 
export OMP_NUM_THREADS=40

source ~/.bashrc

cd matrix\ method

time mpiexec.hydra -genv I_MPI_DEBUG 9 -n $SLURM_NTASKS -genv OMP_NUM_THREADS 40 jupyter nbconvert --execute --to notebook --inplace misc.ipynb
# time mpiexec.hydra -genv I_MPI_DEBUG 9 -n $SLURM_NTASKS -genv OMP_NUM_THREADS 40 /scratch/fizaank.phy21.iitbhu/anaconda3/bin/python optimise.py

mv output_$SLURM_JOB_ID.out "$WORKDIR/output.out"
mv *.npy "$WORKDIR"
mv *.png "$WORKDIR"
mv *$SLURM_JOB_ID.csv "$WORKDIR"

echo "========= Job finished at `date` =========="
