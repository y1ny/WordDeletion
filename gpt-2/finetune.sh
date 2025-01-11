#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES="1"
export NCCL_P2P_DISABLE="1",
export NCCL_IB_DISABLE="1",
# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch; print(torch.cuda.device_count())')
echo $GPUS_PER_NODE
# Number of GPU workers, for single-worker training, please set to 1
NNODES=1
echo $NNODES
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=0
echo $NODE_RANK
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}
echo $MASTER_ADDR
# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}
echo $MASTER_PORT
MODEL="/path/to/gpt2" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See https://qwen.readthedocs.io/en/latest/training/SFT/example.html#data-preparation for more information.
DATA="data/exp1-en/train_data.csv"
EVAL_DATA="data/exp1-en/train_data.csv"
DS_CONFIG_PATH="ds_config_zero3.json"
USE_LORA=False
Q_LORA=False

function usage() {
    echo '
Usage: bash finetune.sh [-m MODEL_PATH] [-d DATA_PATH] [--deepspeed DS_CONFIG_PATH] [--use_lora USE_LORA] [--q_lora Q_LORA]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        --deepspeed )
            shift
            DS_CONFIG_PATH=$1
            ;;
        --use_lora  )
            shift
            USE_LORA=$1
            ;;
        --q_lora    )
            shift
            Q_LORA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

DISTRIBUTED_ARGS="
    --nproc_per_node 1 \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


for s in {1,5,10,50,100,200,500,1000,2000,3000};
do
    for n in {0..9};
    do
        torchrun $DISTRIBUTED_ARGS finetune.py \
            --model_name_or_path $MODEL \
            --data_path $DATA \
            --eval_data_path $EVAL_DATA \
            --bf16 False \
            --output_dir /path/to/model \
            --num_train_epochs 20 \
            --max_steps 500 \
            --n_training_sample $s \
            --n_iter $n \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --gradient_accumulation_steps 2 \
            --evaluation_strategy "steps" \
            --save_strategy "steps" \
            --save_steps 1000 \
            --eval_steps 50 \
            --save_total_limit 5 \
            --learning_rate 2e-6 \
            --weight_decay 0.01 \
            --adam_beta2 0.95 \
            --warmup_ratio 0.01 \
            --from_scratch "without_embedding" \
            --lr_scheduler_type "linear" \
            --logging_steps 10 \
            --report_to "none" \
            --model_max_length 512 \
            --lazy_preprocess True \
            --use_lora ${USE_LORA} \
            --q_lora ${Q_LORA} \
            --gradient_checkpointing
        
        python evaluation.py \
            --n_training_sample $s \
            --n_iter $n \
            --model_path /path/to/model \
            --test_path data/exp1-en/test_sentence.csv \
            --output_path result/exp1-en
    done
done