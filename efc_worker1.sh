#!/usr/bin/bash
#SBATCH --gpus-per-node=1
#SBATCh --nodes=1
#SBATCH --partition=gamerpcs
#SBATCH --nodelist=worker1
#SBATCH --output="log_nair_samp.out"

source /etc/profile.d/modules.sh
module load isfr-advpertbeamf/1.0
srun python /mnt/nfs/efernandez/projects/UNet_Nair/sampling_Nair.py