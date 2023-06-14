#!/bin/bash



OMP_NUM_THREADS=1 torchrun --nnodes=$1 --node_rank=$2 --nproc_per_node=3 train_galai/train.py \
    --model_name_or_path path/to/galactica_base_model/galactica-$3 \
    --data_path  ./OPI_DATA/OPI_nine_tasks_full.json \
    --bf16 True \
    --output_dir path/to/output/galai_ft_opi/galai_ft_opi_full_$3_e$4 \
    --num_train_epochs $4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --tf32 True
