#!/bin/sh

#SBATCH --job-name=BNS-GCN-test
#SBATCH --output=output/FP32-p=0.1-partition=5.txt
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8

# Source the conda script
source /ocean/projects/asc200010p/hliul/miniconda3/etc/profile.d/conda.sh

conda activate BNS-GCN

python main.py \
  --dataset ogbn-products \
  --dropout 0.3 \
  --lr 0.003 \
  --n-partitions 5 \
  --n-epochs 1000 \
  --model graphsage \
  --sampling-rate 0.1 \
  --n-layers 3 \
  --n-hidden 128 \
  --log-every 10 \
  --use-pp \
  --backend gloo