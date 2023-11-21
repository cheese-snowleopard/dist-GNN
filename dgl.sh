#!/bin/sh

#SBATCH --job-name=BNS-GCN-test
#SBATCH --output=dgl.txt
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=8

# Source the conda script
source /ocean/projects/asc200010p/hliul/miniconda3/etc/profile.d/conda.sh

conda activate BNS-GCN

python dgl_primitive.py