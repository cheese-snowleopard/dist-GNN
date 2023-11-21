#!/bin/sh

#SBATCH --job-name=BNS-GCN-test
#SBATCH --output=amp/reddit-full-ddp.txt
#SBATCH --partition=GPU-small
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=8

# Source the conda script
source /ocean/projects/asc200010p/hliul/miniconda3/etc/profile.d/conda.sh

conda activate BNS-GCN


python main.py \
    --dataset reddit \
    --dropout 0.5 \
    --lr 0.01 \
    --n-partitions 2 \
    --n-epochs 1000 \
    --model graphsage \
    --sampling-rate 1.0 \
    --n-layers 4 \
    --n-hidden 256 \
    --log-every 10 \
    --inductive \
    --use-pp \
    --backend gloo

# for N_PARTITIONS in 2 4
# do
#     for SAMPLING_RATE in 1.00 0.10 0.01 0.00
#     do
#         echo -e "\033[1mclean python processes\033[0m"
#         sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
#         echo -e "\033[1m${N_PARTITIONS} partitions, ${SAMPLING_RATE} sampling rate\033[0m"
#         python main.py \
#             --dataset reddit \
#             --dropout 0.5 \
#             --lr 0.01 \
#             --n-partitions ${N_PARTITIONS} \
#             --n-epochs 10 \
#             --model graphsage \
#             --sampling-rate ${SAMPLING_RATE} \
#             --n-layers 4 \
#             --n-hidden 256 \
#             --log-every 10 \
#             --inductive \
#             --use-pp \
#             --backend gloo \
#             --port 13007
#             # |& tee reddit_full/reddit_n${N_PARTITIONS}_p${SAMPLING_RATE}_full.txt
#     done
# done