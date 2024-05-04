#!/usr/bin/bash
#SBATCH --gpus-per-node=1
#SBATCh --nodes=1
#SBATCH --partition=gamerpcs
#SBATCH --nodelist=worker1
#SBATCH --output="log_u.out"

source /etc/profile.d/modules.sh
module load edgar/1.0
srun python /mnt/nfs/efernandez/projects/UNet_Nair/U-NET-BF.py