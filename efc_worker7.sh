#!/usr/bin/bash
#SBATCH --gpus-per-node=1
#SBATCh --nodes=1
#SBATCH --partition=thinkstation-p340
#SBATCH --nodelist=worker7
#SBATCH --output="log_worker7.out"

source /etc/profile.d/modules.sh
module load jesus/1.0
srun python /mnt/nfs/efernandez/projects/UNet_Nair/SUST_new_dataset.py
