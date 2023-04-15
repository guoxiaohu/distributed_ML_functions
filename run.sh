#!/bin/bash -l
#SBATCH --job-name=ai4cfdjob   # Job name
#SBATCH --output=ai4cfd.o%j # Name of stdout output file
#SBATCH --error=ai4cfd.e%j  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes 
#SBATCH --ntasks-per-node=8     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --time=1-12:00:00       # Run time (d-hh:mm:ss)
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH --account=  # Project for billing
##SBATCH --mail-user=username@domain.com


module load cray-python
module load craype-accel-amd-gfx90a
module load rocm

cat << EOF > select_gpu
#!/bin/bash

export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec \$*
EOF

chmod +x ./select_gpu

CPU_BIND="map_cpu:48,56,16,24,1,8,32,40"

export MPICH_GPU_SUPPORT_ENABLED=1

export PYTHONPATH=$HOME/.local:$PYTHONPATH
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH
export CRAY_ACCEL_TARGET=amd_gfx90a

#srun -cpu-bind=${CPU_BIND}  ./select_gpu python parallel_naiver_stokes_3D.py
srun --cpu-bind=${CPU_BIND} ./select_gpu python matmult.py

rm -rf ./select_gpu
