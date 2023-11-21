#!/bin/sh

#SBATCH --job-name=BNS-GCN-test
#SBATCH --output=amp/products-full.txt
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8

# Source the conda script
source /ocean/projects/asc200010p/hliul/miniconda3/etc/profile.d/conda.sh

conda activate BNS-GCN

for N_PARTITIONS in 5 8
do
    for SAMPLING_RATE in 1.00 0.10 0.01 0.00
    do
        echo -e "\033[1mclean python processes\033[0m"
        sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s
        echo -e "\033[1m${N_PARTITIONS} partitions, ${SAMPLING_RATE} sampling rate\033[0m"
        python main.py \
            --dataset ogbn-products \
            --dropout 0.3 \
            --lr 0.003 \
            --n-partitions ${N_PARTITIONS} \
            --n-epochs 500 \
            --model graphsage \
            --sampling-rate ${SAMPLING_RATE} \
            --n-layers 4 \
            --n-hidden 128 \
            --log-every 10 \
            --inductive \
            --use-pp \
            --backend gloo \
            --port 13007 \
            |& tee products_full/reddit_n${N_PARTITIONS}_p${SAMPLING_RATE}_full.txt
    done
done