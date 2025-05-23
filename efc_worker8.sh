#!/usr/bin/bash
#SBATCH --gpus-per-node=1
#SBATCh --nodes=1
#SBATCH --partition=thinkstation-p360
#SBATCH --nodelist=worker8
#SBATCH --output="log_worker8.out"

source /etc/profile.d/modules.sh
module load jesus/1.0
srun python /mnt/nfs/efernandez/projects/UNet_Nair/SUST_new_dataset.py
