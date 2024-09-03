#!/bin/bash
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0    
#SBATCH --mail-user=ytelila@uwo.ca
#SBATCH --mail-type=ALL

cd /home/joet/projects/def-kgroling/joet/fedProto/
module purge

module load python/3.10 scipy-stack
module load arrow/16.1.0

source ~/UWO/bin/activate

python train_iid.py