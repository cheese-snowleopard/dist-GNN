#!/bin/sh

#SBATCH --job-name=BNS-GCN-test
#SBATCH --output=output/FP16-p=0.2-partition=5.txt
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8

# Source the conda script
source /ocean/projects/asc200010p/hliul/miniconda3/etc/profile.d/conda.sh

conda activate BNS-GCN

mkdir results
for N_PARTITIONS in 2 4 8
do
  for SAMPLING_RATE in 0.20 0.02
  do
    echo -e "\033[1mclean python processes\033[0m"
    sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
    echo -e "\033[1m${N_PARTITIONS} partitions, ${SAMPLING_RATE} sampling rate\033[0m"
    python main.py \
      --dataset reddit \
      --dropout 0.5 \
      --lr 0.01 \
      --n-partitions ${N_PARTITIONS} \
      --n-epochs 3000 \
      --model graphsage \
      --sampling-rate ${SAMPLING_RATE} \
      --n-layers 4 \
      --n-hidden 256 \
      --log-every 10 \
      --inductive \
      --use-pp \
      --dtype float16 \
      |& tee results/reddit_n${N_PARTITIONS}_p${SAMPLING_RATE}_fp16.txt
  done
done