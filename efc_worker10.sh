#!/usr/bin/bash
#SBATCH --gpus-per-node=1
#SBATCh --nodes=1
#SBATCH --partition=thinkstation-p360
#SBATCH --nodelist=worker10
#SBATCH --output="log_worker10.out"

source /etc/profile.d/modules.sh
module load students_env/1.0
srun python /mnt/nfs/efernandez/projects/UNet_Nair/computting_metrics_nair.py