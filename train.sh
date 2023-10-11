#!/bin/sh

#SBATCH --job-name=BNS-GCN-test
#SBATCH --output=BNS-GCN-test.txt
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8

# Source the conda script
source /ocean/projects/asc200010p/hliul/miniconda3/etc/profile.d/conda.sh

conda activate BNS-GCN

python main.py \
  --dataset reddit \
  --dropout 0.5 \
  --lr 0.01 \
  --n-partitions 4 \
  --n-epochs 10 \
  --model graphsage \
  --sampling-rate .1 \
  --n-layers 4 \
  --n-hidden 256 \
  --log-every 10 \
  --inductive \
  --use-pp \
  --backend nccl