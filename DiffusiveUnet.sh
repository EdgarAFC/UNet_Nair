#!/usr/bin/bash
#SBATCH --gpus-per-node=1
#SBATCh --nodes=1
#SBATCH --partition=gamerpcs
#SBATCH --nodelist=worker2
#SBATCH --output="log_diff_unet.out"

source /etc/profile.d/modules.sh
module load ifsr-advpertbeamf/1.0
srun python /mnt/nfs/efernandez/projects/UNet_Nair/U-NET-BF.py